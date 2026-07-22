"""使用 PyTRIO 评测 base model 或 OPSD sampler weights 在 AIME 2025 上的表现。

先下载评测集：

uv run python 00-datasets.py --only aime25


Base Model 评测：

uv run python 00-eval-aime25.py \
    --val-n 12 \
    --max-tokens 38912 \
    --temperature 1.0 \
    --enable-thinking false \
    --output eval-results/aime25-base.jsonl

Sample weights 评测：

uv run python 00-eval-aime25.py \
    --val-n 12 \
    --max-tokens 38912 \
    --temperature 1.0 \
    --enable-thinking false \
    --model-path trio://<your_sampler_weights_path> \
    --output eval-results/aime25-sampler-steps25.jsonl

脚本报告 Average@N、Pass@N 和 boxed format rate。AIME 答案均为整数，
因此使用整数精确匹配，不额外依赖 math_verify。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import re
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
import pytrio as trio
from tqdm import tqdm


trio.configure(sampling_timeout=18000,)  # 5 小时，足够评测 30 道题

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = SCRIPT_DIR / "datasets" / "aime_2025"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "eval-results" / "aime25.jsonl"
EXPECTED_ROWS = 30


def parse_bool(value: str) -> bool:
    """把命令行中的 True/False 文本转换为布尔值。"""
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected True or False")


def parse_args() -> argparse.Namespace:
    """解析并校验 AIME25 并发采样与输出相关参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="00-datasets.py 保存的 AIME25 数据目录",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3.5-4B",
        help="PyTRIO sampling client 的基础模型",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="save_weights_for_sampler 返回的 trio:// 路径；留空评测 base model",
    )
    parser.add_argument("--val-n", type=int, default=12, help="每道题生成多少个答案")
    parser.add_argument(
        "--limit", type=int, default=0, help="只评测前 N 题；0 表示全部 30 题"
    )
    parser.add_argument("--concurrency", type=int, default=15, help="并发评测题目数")
    parser.add_argument(
        "--max-tokens", type=int, default=38912, help="每个答案最大生成 token 数"
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--enable-thinking",
        type=parse_bool,
        default=False,
        metavar="{True,False}",
        help="是否启用模型 chat template 的 thinking 模式；默认 False",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="逐题 JSONL 输出；最后一行为 summary",
    )
    args = parser.parse_args()

    for name in ("val_n", "concurrency", "max_tokens"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.temperature < 0:
        raise ValueError("--temperature must be >= 0")
    return args


def load_aime25(path: Path, limit: int) -> Dataset:
    """读取本地 AIME25，验证结构后按需截取少量题目。"""
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 AIME25 数据：{path}\n"
            "请先运行：uv run python 04-opsd/00-datasets.py --only aime25"
        )
    loaded = load_from_disk(str(path))
    dataset = loaded["train"] if isinstance(loaded, DatasetDict) else loaded
    if not isinstance(dataset, Dataset):
        raise TypeError(f"期望 Dataset，实际得到 {type(dataset)!r}")
    missing = sorted({"problem", "answer"} - set(dataset.column_names))
    if missing:
        raise ValueError(
            f"AIME25 缺少字段 {missing}，实际字段为 {dataset.column_names}"
        )
    if len(dataset) != EXPECTED_ROWS:
        raise ValueError(f"AIME25 应有 {EXPECTED_ROWS} 题，实际为 {len(dataset)} 题")
    if limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def extract_last_boxed(text: str) -> str | None:
    """提取最后一个花括号完整闭合的 ``\\boxed{...}``，支持嵌套花括号。

    生成被 max_tokens 截断时，末尾可能只剩半个 ``\\boxed{``，
    此时向前回退到最近一个完整的 ``\\boxed{...}``。
    """
    end = len(text)
    while True:
        start = text.rfind("\\boxed", 0, end)
        if start < 0:
            return None
        left = text.find("{", start)
        if left < 0:
            end = start
            continue
        depth = 0
        for index in range(left, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[left + 1 : index].strip()
        end = start


def normalize_aime_answer(answer: str | None) -> str | None:
    """将 AIME 答案规范为无前导零的整数字符串。"""
    if answer is None:
        return None
    cleaned = answer.strip().replace(",", "").replace("$", "")
    cleaned = re.sub(r"\\(?:text|mathrm)\s*\{([^{}]*)\}", r"\1", cleaned)
    match = re.fullmatch(r"\s*([+-]?\d+)\s*", cleaned)
    if match is None:
        return None
    return str(int(match.group(1)))


def build_prompt_ids(tokenizer: Any, problem: str, enable_thinking: bool) -> list[int]:
    """套用模型 chat template，将一道题编码为采样 prompt token。"""
    content = (
        f"{problem.strip()}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("AIME25 prompt token 为空")
    return prompt_ids


async def evaluate_problem(
    index: int,
    row: dict[str, Any],
    sampling_client: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """并发采样一道题的 N 个答案，并计算正确性与多数投票结果。"""
    problem = str(row["problem"]).strip()
    ground_truth = normalize_aime_answer(str(row["answer"]))
    if ground_truth is None:
        raise ValueError(
            f"AIME25 第 {index} 题 ground truth 不是整数: {row['answer']!r}"
        )

    prompt_ids = build_prompt_ids(tokenizer, problem, args.enable_thinking)
    params = trio.SamplingParams(
        max_tokens=args.max_tokens,
        seed=args.seed + index,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stop=list(
            dict.fromkeys(
                token for token in (tokenizer.eos_token, "<|im_end|>") if token
            )
        ),
    )
    async with semaphore:
        response = await sampling_client.sample_async(
            prompt=trio.ModelInput.from_ints(prompt_ids),
            num_samples=args.val_n,
            sampling_params=params,
            return_text=True,
        )
    if len(response.sequences) != args.val_n:
        raise RuntimeError(
            f"AIME25 第 {index} 题请求 {args.val_n} 条 completion，"
            f"实际返回 {len(response.sequences)} 条"
        )

    generations = []
    for sequence in response.sequences:
        text = sequence.text
        if text is None:
            text = tokenizer.decode(sequence.tokens, skip_special_tokens=False)
        boxed = extract_last_boxed(text)
        predicted = normalize_aime_answer(boxed)
        generations.append(
            {
                "predicted_answer": predicted,
                "boxed_answer": boxed,
                "correct": predicted == ground_truth,
                "formatted": boxed is not None,
                "completion_tokens": len(sequence.tokens),
                "text": text,
            }
        )

    return {
        "type": "problem",
        "problem_id": int(row.get("id", index)),
        "problem": problem,
        "ground_truth": ground_truth,
        "val_n": args.val_n,
        "num_correct": sum(int(item["correct"]) for item in generations),
        "pass_at_n": any(item["correct"] for item in generations),
        "generations": generations,
    }


def summarize(
    results: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    """聚合逐题结果，生成 Average@N、Pass@N 等总体指标。"""
    total_problems = len(results)
    total_generations = sum(len(item["generations"]) for item in results)
    total_correct = sum(item["num_correct"] for item in results)
    total_formatted = sum(
        int(generation["formatted"])
        for item in results
        for generation in item["generations"]
    )
    pass_count = sum(int(item["pass_at_n"]) for item in results)
    return {
        "type": "summary",
        "dataset": "yentinglin/aime_2025",
        "base_model": args.base_model,
        "model_path": args.model_path,
        "enable_thinking": args.enable_thinking,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "val_n": args.val_n,
        "problems": total_problems,
        "generations": total_generations,
        "average_at_n": total_correct / total_generations if total_generations else 0.0,
        "pass_at_n": pass_count / total_problems if total_problems else 0.0,
        "format_rate": total_formatted / total_generations
        if total_generations
        else 0.0,
        "correct_generations": total_correct,
        "passed_problems": pass_count,
    }


async def evaluate(args: argparse.Namespace) -> None:
    """创建 PyTRIO 采样客户端，并发完成评测和结果落盘。"""
    dataset = load_aime25(args.dataset_path, args.limit)
    service_client = trio.ServiceClient()
    sampling_client = await service_client.create_sampling_client_async(
        base_model=args.base_model,
        model_path=args.model_path,
    )
    tokenizer = sampling_client.get_tokenizer()
    semaphore = asyncio.Semaphore(args.concurrency)

    with tqdm(total=len(dataset), desc="AIME25", unit="problem") as progress:

        async def evaluate_and_track(index: int, row: dict[str, Any]) -> dict[str, Any]:
            """评测单题并在完成后安全更新题目级进度条。"""
            result = await evaluate_problem(
                index,
                row,
                sampling_client,
                tokenizer,
                args,
                semaphore,
            )
            progress.update(1)
            return result

        results = await asyncio.gather(
            *(evaluate_and_track(index, row) for index, row in enumerate(dataset))
        )

    summary = summarize(results, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
        file.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(
        f"AIME25 Average@{args.val_n}: {summary['average_at_n']:.2%} | "
        f"Pass@{args.val_n}: {summary['pass_at_n']:.2%} | "
        f"Format: {summary['format_rate']:.2%}"
    )
    print("Evaluation summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {args.output}")


if __name__ == "__main__":
    asyncio.run(evaluate(parse_args()))
