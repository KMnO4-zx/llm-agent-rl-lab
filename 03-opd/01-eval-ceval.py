"""C-Eval 中文通用能力评测脚本。

使用 TRIO OpenAI-compatible API 评测模型。
`--model` 可以传基模名称，也可以传 trio:// sampler 权重路径。

启动命令：
uv run python 01-eval-ceval.py \
    --model Qwen/Qwen3.5-4B \
    --limit 10 \
    --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = SCRIPT_DIR / "datasets" / "ceval-non-med" / "test_sample.jsonl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "eval-results"
DEFAULT_BASE_URL = "https://pytrio.cn/api/openai/v1"
CHOICE_SYSTEM_MESSAGE = (
    "你是中文单项选择题作答助手。请在内部完成必要推理，"
    "但最终回答只能包含 A、B、C、D 中的一个大写字母，"
    "不要输出推理过程、解释、标点或其他文字。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C-Eval 中文通用能力评测")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH, help="C-Eval JSONL 文件")
    parser.add_argument("--output-path", type=Path, default=None, help="预测结果 JSONL 输出路径；不传则按 model 自动生成")
    parser.add_argument("--metrics-path", type=Path, default=None, help="准确率指标 JSON 输出路径；默认根据 output-path 自动生成")
    parser.add_argument("--limit", type=int, default=0, help="最多评测多少条；<=0 表示全量")

    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible API base_url")
    parser.add_argument("--api-key-env", default="PYTRIO_API_KEY", help="读取 API key 的环境变量名")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B", help="模型名或 trio:// sampler 权重路径")
    parser.add_argument("--max-tokens", type=int, default=4096, help="每题最多生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.01, help="采样 temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="采样 top_p")
    parser.add_argument("--concurrency", type=int, default=4, help="并发请求数量")
    parser.add_argument("--timeout", type=float, default=120.0, help="单次请求超时时间，单位秒")
    parser.add_argument("--show-errors", type=int, default=5, help="终端最多展示多少条错误样例")
    args = parser.parse_args()

    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")
    return args


def model_slug(model: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model.strip()).strip("-").lower()
    return slug or "model"


def default_output_path(model: str) -> Path:
    return DEFAULT_OUTPUT_DIR / f"ceval_eval_{model_slug(model)}.jsonl"


def load_examples(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到评测文件：{path}，请先运行 00-download-dataset.py")

    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "answer_idx" not in row:
                raise ValueError(f"评测样本缺少 answer_idx 字段：{row.keys()}")
            examples.append(row)
            if limit > 0 and len(examples) >= limit:
                break
    return examples


def build_prompt(row: dict[str, Any]) -> str:
    question = str(row.get("question", "")).strip()
    options = row.get("options") or {}
    if question and all(key in options for key in ("A", "B", "C", "D")):
        return "\n".join(
            [
                "以下是中国考试中的单项选择题。请仔细思考，并只输出最终答案选项字母。",
                "",
                f"题目：{question}",
                f"A. {options['A']}",
                f"B. {options['B']}",
                f"C. {options['C']}",
                f"D. {options['D']}",
                "",
                "答案：",
            ]
        )
    if "prompt" in row:
        return str(row["prompt"])
    raise ValueError(f"评测样本缺少 question/options 或 prompt 字段：{row.keys()}")


def create_client(args: argparse.Namespace) -> AsyncOpenAI:
    load_dotenv(SCRIPT_DIR / ".env", encoding="utf-8-sig", override=True)
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"请在 03-opd/.env 或环境变量中设置 {args.api_key_env}")

    return AsyncOpenAI(
        base_url=args.base_url,
        api_key=api_key,
        timeout=args.timeout,
    )


async def infer_one(client: AsyncOpenAI, args: argparse.Namespace, prompt: str) -> str:
    response = await client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": CHOICE_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    return response.choices[0].message.content or ""


def parse_choice(text: str) -> str | None:
    raw_text = text.strip()
    if "</think>" in raw_text:
        segments = [raw_text.rsplit("</think>", 1)[-1]]
    else:
        segments = [raw_text]

    for segment in segments:
        normalized = segment.strip().upper()
        if not normalized:
            continue
        exact_match = re.fullmatch(r"[^A-Z]*([ABCD])[^A-Z]*", normalized)
        if exact_match:
            return exact_match.group(1)

        patterns = [
            r"(?:FINAL\s*(?:ANSWER|OUTPUT)|ANSWER|OUTPUT|最终答案|正确答案|答案|选项|选择|应选|故选)\s*(?:IS|是|为|:|：|->)?\s*[\(\[【]?\s*([ABCD])",
            r"[\(\[【]\s*([ABCD])\s*[\)\]】]",
        ]
        candidates: list[str] = []
        for pattern in patterns:
            candidates.extend(re.findall(pattern, normalized))
        if candidates:
            return candidates[-1]

        for line in reversed(normalized.splitlines()):
            line_match = re.fullmatch(r"\s*([ABCD])\s*[\.。:：、]?\s*", line)
            if line_match:
                return line_match.group(1)

    fallback = re.findall(r"(?<![A-Z])([ABCD])(?![A-Z])", segments[0].upper()[-1000:])
    if fallback:
        return fallback[-1]
    return None


async def evaluate_one(
    index: int,
    row: dict[str, Any],
    client: AsyncOpenAI,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    prompt = build_prompt(row)
    async with semaphore:
        output = await infer_one(client, args, prompt)

    prediction = parse_choice(output)
    answer = str(row["answer_idx"]).strip().upper()
    is_correct = prediction == answer
    return {
        "index": index,
        "row_id": row.get("row_id"),
        "subject": row.get("subject"),
        "source_split": row.get("source_split"),
        "question": row.get("question"),
        "answer_idx": answer,
        "prediction": prediction,
        "correct": is_correct,
        "output": output,
        "prompt": prompt,
        "model": args.model,
    }


def subject_metrics(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("subject") or "unknown")].append(record)

    metrics = {}
    for subject, items in sorted(grouped.items()):
        total = len(items)
        correct = sum(1 for item in items if item["correct"])
        invalid = sum(1 for item in items if item["prediction"] is None)
        metrics[subject] = {
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "invalid": invalid,
        }
    return metrics


async def evaluate(args: argparse.Namespace) -> None:
    examples = load_examples(args.dataset_path, args.limit)
    if not examples:
        raise ValueError(f"评测文件为空：{args.dataset_path}")

    args.output_path = args.output_path or default_output_path(args.model)
    client = create_client(args)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(examples)} C-Eval examples")
    print(f"API base_url: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"Concurrency: {args.concurrency}")

    start = time.time()
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(evaluate_one(index, row, client, args, semaphore))
        for index, row in enumerate(examples)
    ]
    records: list[dict[str, Any]] = []
    try:
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="C-Eval eval", unit="item"):
            records.append(await future)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.close()

    records.sort(key=lambda item: item["index"])
    correct = sum(1 for item in records if item["correct"])
    invalid = sum(1 for item in records if item["prediction"] is None)
    errors = [item for item in records if not item["correct"]][: args.show_errors]

    with args.output_path.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(examples)
    accuracy = correct / total if total else 0.0
    elapsed = time.time() - start
    by_subject = subject_metrics(records)
    macro_accuracy = (
        sum(item["accuracy"] for item in by_subject.values()) / len(by_subject)
        if by_subject
        else 0.0
    )
    metrics = {
        "accuracy": accuracy,
        "macro_subject_accuracy": macro_accuracy,
        "correct": correct,
        "total": total,
        "invalid": invalid,
        "elapsed_seconds": elapsed,
        "model": args.model,
        "concurrency": args.concurrency,
        "dataset_path": str(args.dataset_path),
        "output_path": str(args.output_path),
        "by_subject": by_subject,
    }
    metrics_path = args.metrics_path or args.output_path.with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"accuracy {accuracy:.4f} | macro_subject_accuracy {macro_accuracy:.4f} | "
        f"correct {correct}/{total} | invalid {invalid} | time {elapsed:.2f}s"
    )
    print(f"Saved predictions: {args.output_path}")
    print(f"Saved metrics: {metrics_path}")

    if errors:
        print("\n错误样例：")
        for item in errors:
            question = str(item["question"])[:80]
            print(
                f"- #{item['index']} subject={item['subject']} pred={item['prediction']} "
                f"answer={item['answer_idx']} question={question}"
            )


def main() -> None:
    asyncio.run(evaluate(parse_args()))


if __name__ == "__main__":
    main()
