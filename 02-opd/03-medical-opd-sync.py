"""同步版医疗 OPD：medical SFT 4B teacher -> fresh 4B base student。

本脚本和 02-medical-sft.py 使用同一份 medical-o1-reasoning-SFT-zh 数据：
student 只读取 question 并按当前策略采样，SFT teacher 对 student 的实际轨迹计算
逐 token logprob，再用 reverse KL advantage 更新 fresh LoRA student。

MedQA-ZH 和 C-Eval 都不进入训练。它们只用于评测 step checkpoint：
- MedQA-ZH @ max_tokens=1024 检查短输出预算下的医疗表现；
- C-Eval 检查通用能力是否下降。

小成本试跑：
uv run python 03-medical-opd-sync.py \
    --teacher-model-path YOUR_SFT_SAMPLER_WEIGHTS_PATH \
    --steps 10 \
    --batch-size 4 \
    --group-size 4 \
    --sample-size 100 \
    --max-tokens 2048 \
    --save-every-steps 5 \
    --swanlab-mode disabled

第一轮正式试验：
uv run python 03-medical-opd-sync.py \
    --teacher-model-path YOUR_SFT_SAMPLER_WEIGHTS_PATH \
    --steps 300 \
    --batch-size 4 \
    --group-size 4 \
    --sample-size 0 \
    --max-tokens 2048 \
    --learning-rate 4e-5 \
    --save-every-steps 300 \
    --swanlab-mode online

注意：`--max-tokens 2048` 是 completion 输出预算，不是总上下文长度。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytrio as trio
import swanlab
from tqdm import tqdm


trio.configure(timeout=600)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = (
    SCRIPT_DIR / "datasets" / "medical-o1-reasoning-SFT-zh" / "train.jsonl"
)
DEFAULT_SYSTEM_MESSAGE = "你是一个中文医疗问答助手。请根据题目给出严谨的医学推理和最终答案。"
LOSS_FNS = ("importance_sampling", "ppo")


@dataclass(frozen=True)
class PromptRollout:
    datums: list[trio.Datum]
    reverse_kls: list[float]
    completion_token_counts: list[int]


class ChineseArgumentParser(argparse.ArgumentParser):
    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "用法:", 1)

    def format_help(self) -> str:
        return super().format_help().replace("usage:", "用法:", 1)


def model_slug(base_model: str) -> str:
    name = base_model.rsplit("/", 1)[-1].lower().replace("qwen3.5", "qwen35")
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def dataset_slug(path: Path) -> str:
    name = path.parent.name.lower().replace("_", "-")
    return re.sub(r"[^a-z0-9-]+", "-", name).strip("-")


def build_run_name(args: argparse.Namespace) -> str:
    loss_slug = args.loss_fn.replace("_", "-")
    steps_slug = "full" if args.steps == 0 else f"steps{args.steps}"
    return (
        f"opd-medical-sync-{model_slug(args.base_model)}-"
        f"{dataset_slug(args.dataset_path)}-{loss_slug}-{steps_slug}"
    )


def parse_args() -> argparse.Namespace:
    parser = ChineseArgumentParser(
        description="PyTRIO 同步版医疗 OPD：SFT teacher -> fresh base student",
        add_help=False,
    )
    parser._optionals.title = "选项"
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="medical-o1-reasoning-SFT-zh train JSONL",
    )
    parser.add_argument("--sample-size", type=int, default=0, help="随机抽样 prompt 数；0 表示全用")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B", help="fresh LoRA student 的基础模型")
    parser.add_argument("--lora-rank", type=int, default=16, help="fresh student 的 LoRA rank，范围 4-64")
    parser.add_argument(
        "--teacher-base-model",
        default=None,
        help="SFT teacher 的基础模型；默认和 student base 相同",
    )
    parser.add_argument(
        "--teacher-model-path",
        required=True,
        help="medical SFT 的 trio:// sampler_weights 路径",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="训练 step 数；0 表示完整遍历当前数据一次",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="每个 step 使用的医疗 prompt 数")
    parser.add_argument("--group-size", type=int, default=4, help="每个 prompt 的 student completion 数")
    parser.add_argument("--max-tokens", type=int, default=2048, help="每条 completion 最多生成 token 数")
    parser.add_argument("--temperature", type=float, default=1.0, help="student rollout temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="student rollout top_p")
    parser.add_argument("--top-k", type=int, default=-1, help="student rollout top_k；-1 表示不限制")
    parser.add_argument(
        "--system-message",
        default=DEFAULT_SYSTEM_MESSAGE,
        help="与 medical SFT 一致的 system message；空字符串表示不加",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="chat template 是否启用 thinking",
    )
    parser.add_argument("--kl-penalty-coef", type=float, default=1.0, help="reverse KL advantage 系数")
    parser.add_argument("--learning-rate", type=float, default=4e-5, help="Adam 学习率")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2")
    parser.add_argument(
        "--sampler-refresh-steps",
        type=int,
        default=1,
        help="每隔多少 step 用最新 student 权重刷新 sampler",
    )
    parser.add_argument("--loss-fn", choices=LOSS_FNS, default="ppo", help="PyTRIO 内置 RL loss")
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=50,
        help="每多少 step 保存一次 state + sampler weights；0 表示不保存中间 checkpoint，最终 step 仍保存",
    )
    parser.add_argument("--swanlab", action=argparse.BooleanOptionalAction, default=True, help="是否启用 SwanLab")
    parser.add_argument("--swanlab-project", default="llm-agent-rl-lab", help="SwanLab 项目名")
    parser.add_argument("--swanlab-workspace", default=None, help="SwanLab workspace")
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "offline", "disabled"],
        default=None,
        help="SwanLab 运行模式",
    )
    args = parser.parse_args()

    for name in ("batch_size", "group_size", "max_tokens", "sampler_refresh_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.steps < 0:
        raise ValueError("--steps must be >= 0")
    if args.sample_size < 0:
        raise ValueError("--sample-size must be >= 0")
    if not 4 <= args.lora_rank <= 64:
        raise ValueError("--lora-rank must be between 4 and 64")
    if args.kl_penalty_coef <= 0:
        raise ValueError("--kl-penalty-coef must be > 0")
    if args.save_every_steps < 0:
        raise ValueError("--save-every-steps must be >= 0")

    args.teacher_base_model = args.teacher_base_model or args.base_model
    args.run_name = build_run_name(args)
    return args


def load_medical_prompts(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.dataset_path.exists():
        raise FileNotFoundError(
            f"找不到 medical-o1 数据：{args.dataset_path}，请先运行 00-download-dataset.py"
        )

    rows: list[dict[str, str]] = []
    with args.dataset_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            question = str(row.get("question", "")).strip()
            if not question:
                raise ValueError(f"medical-o1 第 {line_number} 行缺少 question")
            rows.append({"question": question})

    if not rows:
        raise ValueError(f"medical-o1 数据为空：{args.dataset_path}")
    random.Random(args.seed).shuffle(rows)
    if args.sample_size > 0:
        rows = rows[: min(args.sample_size, len(rows))]
    return rows


def build_prompt_tokens(
    tokenizer: Any,
    question: str,
    system_message: str,
    enable_thinking: bool,
) -> list[int]:
    messages = []
    if system_message.strip():
        messages.append({"role": "system", "content": system_message.strip()})
    messages.append({"role": "user", "content": question.strip()})
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("Prompt tokens are empty")
    return prompt_ids


def completion_teacher_logprobs(
    teacher_client: Any,
    prompt_ids: list[int],
    completion_ids: list[int],
) -> list[float]:
    all_ids = prompt_ids + completion_ids
    all_logprobs = teacher_client.compute_logprobs(
        trio.ModelInput.from_ints(all_ids)
    ).result()
    completion_logprobs = all_logprobs[len(prompt_ids) :]
    if len(completion_logprobs) != len(completion_ids) or any(
        value is None for value in completion_logprobs
    ):
        raise ValueError("Invalid teacher logprobs for completion tokens")
    return [float(value) for value in completion_logprobs]


def build_opd_datum(
    prompt_ids: list[int],
    completion_ids: list[int],
    old_logprobs: list[float],
    advantages: list[float] | np.ndarray,
) -> trio.Datum:
    prompt_loss_len = len(prompt_ids) - 1
    input_ids = prompt_ids + completion_ids[:-1]
    target_ids = [0] * prompt_loss_len + completion_ids
    padded_logprobs = [0.0] * prompt_loss_len + old_logprobs
    padded_advantages = [0.0] * prompt_loss_len + list(advantages)
    if not (
        len(input_ids)
        == len(target_ids)
        == len(padded_logprobs)
        == len(padded_advantages)
    ):
        raise ValueError("OPD datum fields must have the same length")
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(input_ids),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_ids, dtype=np.int64),
            "logprobs": np.asarray(padded_logprobs, dtype=np.float32),
            "advantages": np.asarray(padded_advantages, dtype=np.float32),
        },
    )


def run_prompt_rollout(
    student_sampler: Any,
    teacher_client: Any,
    tokenizer: Any,
    row: dict[str, str],
    args: argparse.Namespace,
    sampling_params: trio.SamplingParams,
) -> PromptRollout:
    prompt_ids = build_prompt_tokens(
        tokenizer=tokenizer,
        question=row["question"],
        system_message=args.system_message,
        enable_thinking=args.enable_thinking,
    )
    sample_result = student_sampler.sample(
        prompt=trio.ModelInput.from_ints(prompt_ids),
        num_samples=args.group_size,
        sampling_params=sampling_params,
        return_text=False,
    ).result()

    datums: list[trio.Datum] = []
    reverse_kls: list[float] = []
    completion_token_counts: list[int] = []
    for sequence in sample_result.sequences:
        completion_ids = list(sequence.tokens)
        if not completion_ids:
            continue
        student_logprobs = [float(value) for value in sequence.logprobs]
        if len(student_logprobs) != len(completion_ids):
            raise ValueError("Student token/logprob length mismatch")
        teacher_logprobs = completion_teacher_logprobs(
            teacher_client=teacher_client,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
        )
        reverse_kl = np.asarray(student_logprobs) - np.asarray(teacher_logprobs)
        advantages = -args.kl_penalty_coef * reverse_kl
        datums.append(
            build_opd_datum(
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                old_logprobs=student_logprobs,
                advantages=advantages,
            )
        )
        reverse_kls.extend(reverse_kl.tolist())
        completion_token_counts.append(len(completion_ids))

    return PromptRollout(
        datums=datums,
        reverse_kls=reverse_kls,
        completion_token_counts=completion_token_counts,
    )


def batch_for_step(
    rows: list[dict[str, str]],
    step: int,
    batch_size: int,
    full_dataset_run: bool,
) -> list[dict[str, str]]:
    if full_dataset_run:
        start = step * batch_size
        return rows[start : start + batch_size]
    return [rows[(step * batch_size + offset) % len(rows)] for offset in range(batch_size)]


def pick_loss_metric(metrics: dict[str, float]) -> float | None:
    for key in (
        "trainer/loss",
        "trainer/loss_mean",
        "trainer/loss/mean",
        "trainer/policy_loss",
        "trainer/ppo_loss",
    ):
        if key in metrics:
            return float(metrics[key])
    for key, value in metrics.items():
        if "loss" in key.lower():
            return float(value)
    return None


def start_swanlab(
    args: argparse.Namespace,
    dataset_size: int,
    total_steps: int,
) -> Any | None:
    if not args.swanlab:
        return None
    config = vars(args).copy()
    config["dataset_path"] = str(args.dataset_path)
    config["dataset_size"] = dataset_size
    config["total_steps"] = total_steps
    return swanlab.init(
        project=args.swanlab_project,
        name=args.run_name,
        workspace=args.swanlab_workspace,
        mode=args.swanlab_mode,
        config=config,
        tags=["PyTrio", "OPD", "medical", "SFT-teacher", "sync"],
        log_dir=str(SCRIPT_DIR / "swanlog"),
    )


def save_checkpoint(
    training_client: Any,
    swanlab_run: Any | None,
    args: argparse.Namespace,
    completed_steps: int,
) -> tuple[str, str]:
    tag = f"step{completed_steps:06d}"
    state_name = f"{args.run_name}-{tag}"
    weights_name = f"{args.run_name}-{tag}"
    state_result = training_client.save_state(state_name).result()
    weights_result = training_client.save_weights_for_sampler(weights_name).result()
    print(f"Saved checkpoint [{tag}] state: {state_result.path}")
    print(f"Saved checkpoint [{tag}] weights: {weights_result.path}")
    if swanlab_run is not None:
        swanlab.log(
            {
                f"save/{tag}_state_path": swanlab.Text(state_result.path),
                f"save/{tag}_weights_path": swanlab.Text(weights_result.path),
            },
            step=completed_steps,
        )
    return state_result.path, weights_result.path


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    rows = load_medical_prompts(args)
    full_dataset_run = args.steps == 0
    total_steps = (
        (len(rows) + args.batch_size - 1) // args.batch_size
        if full_dataset_run
        else args.steps
    )
    print(f"Loaded medical-o1 prompts: {len(rows)}")
    print(f"Training steps: {total_steps}")
    print(f"Run name: {args.run_name}")
    print("C-Eval is evaluation-only and is not loaded by this training script.")

    service_client = trio.ServiceClient()
    swanlab_run = None
    try:
        print(f"Creating fresh LoRA student from base: {args.base_model}")
        training_client = service_client.create_lora_training_client(
            base_model=args.base_model,
            rank=args.lora_rank,
            seed=args.seed,
        )
        teacher_client = service_client.create_sampling_client(
            base_model=args.teacher_base_model,
            model_path=args.teacher_model_path,
        )
        # SamplingClient 按完整模型名加载 tokenizer，可避免 TrainingClient 在部分
        # SDK 版本里只识别内部模型别名的问题。
        tokenizer = teacher_client.get_tokenizer()
        sampling_params = trio.SamplingParams(
            max_tokens=args.max_tokens,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stop=list(
                dict.fromkeys(
                    token
                    for token in (tokenizer.eos_token, "<|im_end|>")
                    if token
                )
            ),
        )
        adam = trio.AdamParams(
            learning_rate=args.learning_rate,
            beta1=args.beta1,
            beta2=args.beta2,
        )
        swanlab_run = start_swanlab(args, len(rows), total_steps)
        student_sampler = None

        for step in range(total_steps):
            step_start = time.time()
            if student_sampler is None or step % args.sampler_refresh_steps == 0:
                student_sampler = training_client.save_weights_and_get_sampling_client()

            batch = batch_for_step(
                rows,
                step,
                args.batch_size,
                full_dataset_run,
            )
            rollouts = [
                run_prompt_rollout(
                    student_sampler=student_sampler,
                    teacher_client=teacher_client,
                    tokenizer=tokenizer,
                    row=row,
                    args=args,
                    sampling_params=sampling_params,
                )
                for row in tqdm(batch, desc=f"Medical OPD step {step}", unit="prompt")
            ]
            datums = [datum for rollout in rollouts for datum in rollout.datums]
            reverse_kls = [value for rollout in rollouts for value in rollout.reverse_kls]
            completion_counts = [
                value
                for rollout in rollouts
                for value in rollout.completion_token_counts
            ]
            if not datums:
                raise RuntimeError("No OPD datums were built")

            fwd_bwd = training_client.forward_backward(datums, loss_fn=args.loss_fn)
            optim = training_client.optim_step(adam)
            fwd_bwd_result = fwd_bwd.result()
            optim.result()

            completed_steps = step + 1
            elapsed = time.time() - step_start
            completion_tokens_total = int(sum(completion_counts))
            metrics: dict[str, float | int] = {
                "data/prompts": len(batch),
                "data/datums": len(datums),
                "data/completion_tokens_mean": float(np.mean(completion_counts)),
                "data/completion_tokens_total": completion_tokens_total,
                "opd/reverse_kl_mean": float(np.mean(reverse_kls)),
                "opd/reverse_kl_std": float(np.std(reverse_kls)),
                "train/step": completed_steps,
                "train/learning_rate": args.learning_rate,
                "time/step_elapsed_time": elapsed,
                "step/completion_tokens_per_second": completion_tokens_total / elapsed,
            }
            metrics.update(
                {
                    f"trainer/{key}": float(value)
                    for key, value in dict(fwd_bwd_result.metrics).items()
                }
            )
            loss_value = pick_loss_metric(
                {key: float(value) for key, value in metrics.items()}
            )
            if loss_value is not None:
                metrics["train/loss"] = loss_value
            if swanlab_run is not None:
                swanlab.log(metrics, step=completed_steps)

            loss_text = "n/a" if loss_value is None else f"{loss_value:.4f}"
            print(
                f"step {completed_steps:03d}/{total_steps} | datums {len(datums)} | "
                f"completion tokens mean {metrics['data/completion_tokens_mean']:.1f} | "
                f"reverse_kl {metrics['opd/reverse_kl_mean']:.4f} | "
                f"tokens/s {metrics['step/completion_tokens_per_second']:.1f} | "
                f"loss {loss_text} | time {elapsed:.2f}s"
            )

            should_save = completed_steps == total_steps or (
                args.save_every_steps > 0
                and completed_steps % args.save_every_steps == 0
            )
            if should_save:
                save_checkpoint(
                    training_client=training_client,
                    swanlab_run=swanlab_run,
                    args=args,
                    completed_steps=completed_steps,
                )
    finally:
        if swanlab_run is not None:
            swanlab.finish()


if __name__ == "__main__":
    started_at = time.time()
    main(parse_args())
    print("#" * 50)
    print("# all done")
    print(f"# train cost {time.time() - started_at:.2f}s")
    print("#" * 50)
