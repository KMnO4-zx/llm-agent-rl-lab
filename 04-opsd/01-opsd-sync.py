"""PyTRIO 同步版 OPSD：固定 base teacher + privileged reference solution。

这个版本按数据流顺序展开，便于理解：

1. student 只看 ``problem``，用当前 LoRA 策略采样 completion；
2. fixed teacher 使用同一 base model，但额外看见 ``solution``；
3. teacher 对 student 的同一条 completion 计算逐 token logprob；
4. ``advantage = teacher_logprob - student_logprob``；
5. PyTRIO ``importance_sampling`` 更新 student。

测试:
uv run python 01-opsd-sync.py \
    --steps 10 \
    --batch-size 4 \
    --group-size 1 \
    --max-tokens 1024 \
    --sample-size 100 \
    --save-every-steps 10 \
    --swanlab-mode disabled

默认每 25 step 同时保存可续训 state 和可评测 sampler weights。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import re
import time
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
import numpy as np
import pytrio as trio
import swanlab
from tqdm import tqdm


trio.configure(timeout=1800)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = SCRIPT_DIR / "datasets" / "openthoughts_math_30k_opsd"
EXPECTED_DATASET_ROWS = 29_434

STUDENT_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
TEACHER_TRANSITION = (
    "After reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
    "own words and independent reasoning, derive the same final answer to the problem above. "
    "Think step by step, explore different approaches, and don't be afraid to backtrack "
    "or reconsider if something doesn't work out:"
)


@dataclass(frozen=True)
class PromptRollout:
    datums: list[trio.Datum]
    reverse_kls: list[float]
    student_logprobs: list[float]
    teacher_logprobs: list[float]
    completion_token_counts: list[int]
    sample_text: str | None


def model_slug(base_model: str) -> str:
    """把基础模型名称转换为适合放进实验名的短标识。"""
    name = base_model.rsplit("/", 1)[-1].lower().replace("qwen3.5", "qwen35")
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def default_run_name(args: argparse.Namespace) -> str:
    """根据模型、目标函数和步数生成默认实验名称。"""
    steps = "full" if args.steps == 0 else f"steps{args.steps}"
    return f"opsd-sync-{model_slug(args.base_model)}-sampled-token-{steps}"


def parse_args() -> argparse.Namespace:
    """解析并校验同步 OPSD 的训练、采样、保存和日志参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="00-datasets.py 保存的 Openthoughts_math_30k_opsd 目录",
    )
    parser.add_argument(
        "--sample-size", type=int, default=0, help="随机抽样题数；0 表示全量"
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument(
        "--train-unembed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="LoRA 是否训练 unembed；论文只训练 attention + MLP，默认 False",
    )
    parser.add_argument(
        "--steps", type=int, default=100, help="训练 step；0 表示遍历当前数据一次"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="每个 step 的 problem 数"
    )
    parser.add_argument(
        "--group-size", type=int, default=1, help="每个 problem 的 completion 数"
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--student-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="student rollout 是否启用 thinking；算法复现默认 False",
    )
    parser.add_argument(
        "--teacher-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="privileged teacher prompt 是否启用 thinking；算法复现默认 False",
    )
    parser.add_argument("--kl-penalty-coef", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument(
        "--sampler-refresh-steps",
        type=int,
        default=1,
        help="刷新 student sampler 的间隔；1 才是严格 on-policy",
    )
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=25,
        help="每 N step 保存 state + sampler weights；0 仅保存最终 checkpoint",
    )
    parser.add_argument(
        "--log-sample-every-steps",
        type=int,
        default=10,
        help="每 N step 向 SwanLab 记录一个 student completion；0 表示不记录文本",
    )
    parser.add_argument(
        "--run-name", default=None, help="SwanLab 和 TRIO checkpoint 名称前缀"
    )

    parser.add_argument(
        "--swanlab", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--swanlab-project", default="llm-agent-rl-lab-opsd")
    parser.add_argument("--swanlab-workspace", default=None)
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "offline", "disabled"],
        default=None,
    )
    args = parser.parse_args()

    for name in ("batch_size", "group_size", "max_tokens", "sampler_refresh_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    for name in ("steps", "sample_size", "save_every_steps", "log_sample_every_steps"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
    if not 4 <= args.lora_rank <= 64:
        raise ValueError("--lora-rank must be between 4 and 64")
    if args.kl_penalty_coef <= 0:
        raise ValueError("--kl-penalty-coef must be > 0")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be > 0")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if args.temperature < 0:
        raise ValueError("--temperature must be >= 0")

    args.run_name = args.run_name or default_run_name(args)
    return args


def load_training_dataset(args: argparse.Namespace) -> Dataset:
    """读取本地 OPSD 数据，校验字段和行数后打乱或抽样。"""
    if not args.dataset_path.exists():
        raise FileNotFoundError(
            f"找不到 OPSD 数据：{args.dataset_path}\n"
            "请先运行：uv run python 04-opsd/00-datasets.py --only opsd"
        )
    loaded = load_from_disk(str(args.dataset_path))
    dataset = loaded["train"] if isinstance(loaded, DatasetDict) else loaded
    if not isinstance(dataset, Dataset):
        raise TypeError(f"期望 Dataset，实际得到 {type(dataset)!r}")
    missing = sorted({"problem", "solution"} - set(dataset.column_names))
    if missing:
        raise ValueError(
            f"OPSD 数据缺少字段 {missing}；实际字段为 {dataset.column_names}"
        )
    if len(dataset) != EXPECTED_DATASET_ROWS:
        raise ValueError(
            f"官方 OPSD 数据应有 {EXPECTED_DATASET_ROWS:,} 条，实际为 {len(dataset):,} 条"
        )

    dataset = dataset.shuffle(seed=args.seed)
    if args.sample_size > 0:
        dataset = dataset.select(range(min(args.sample_size, len(dataset))))
    if not dataset:
        raise ValueError("训练数据为空")
    return dataset


def batch_for_step(
    dataset: Dataset,
    step: int,
    batch_size: int,
    full_dataset_run: bool,
) -> Dataset:
    """按当前 step 选择一个 batch，固定步数模式下允许循环取样。"""
    start = step * batch_size
    if full_dataset_run:
        indices = list(range(start, min(start + batch_size, len(dataset))))
    else:
        indices = [(start + offset) % len(dataset) for offset in range(batch_size)]
    return dataset.select(indices)


def render_chat_prompt(
    tokenizer: Any,
    user_message: str,
    enable_thinking: bool,
) -> list[int]:
    """应用 chat template 并把用户消息编码为 prompt token。"""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("chat prompt token 为空")
    return prompt_ids


def build_student_prompt_ids(
    tokenizer: Any, problem: str, enable_thinking: bool
) -> list[int]:
    """构造只包含题目、不包含参考解答的 Student prompt。"""
    user_message = f"Problem: {problem.strip()}\n\n{STUDENT_INSTRUCTION}"
    return render_chat_prompt(tokenizer, user_message, enable_thinking)


def build_teacher_prompt_ids(
    tokenizer: Any,
    problem: str,
    solution: str,
    enable_thinking: bool,
) -> list[int]:
    """构造包含题目和特权参考解答的 Teacher prompt。"""
    user_message = (
        f"Problem: {problem.strip()}\n\n"
        "Here is a reference solution to this problem:\n"
        "=== Reference Solution Begin ===\n"
        f"{solution.strip()}\n"
        "=== Reference Solution End ===\n\n\n"
        f"{TEACHER_TRANSITION}\n\n"
        f"{STUDENT_INSTRUCTION}"
    )
    return render_chat_prompt(tokenizer, user_message, enable_thinking)


def teacher_completion_logprobs(
    teacher_client: Any,
    teacher_prompt_ids: list[int],
    completion_ids: list[int],
) -> list[float]:
    """计算 Teacher 对 Student 实际 completion 的逐 token logprob。"""
    all_ids = teacher_prompt_ids + completion_ids
    all_logprobs = teacher_client.compute_logprobs(
        trio.ModelInput.from_ints(all_ids)
    ).result()
    completion_logprobs = all_logprobs[len(teacher_prompt_ids) :]
    if len(completion_logprobs) != len(completion_ids):
        raise ValueError(
            "Teacher token/logprob 长度不一致："
            f"{len(completion_ids)} != {len(completion_logprobs)}"
        )
    if any(value is None for value in completion_logprobs):
        raise ValueError("Teacher completion logprob 中存在 None")
    return [float(value) for value in completion_logprobs]


def build_opd_datum(
    student_prompt_ids: list[int],
    completion_ids: list[int],
    old_logprobs: list[float],
    advantages: np.ndarray,
) -> trio.Datum:
    """右移并对齐轨迹字段，构造 importance_sampling 所需 Datum。"""
    if not completion_ids:
        raise ValueError("completion 不能为空")
    if len(completion_ids) != len(old_logprobs) or len(completion_ids) != len(
        advantages
    ):
        raise ValueError("completion、old_logprobs、advantages 长度必须一致")

    prompt_loss_len = len(student_prompt_ids) - 1
    input_ids = student_prompt_ids + completion_ids[:-1]
    target_ids = [0] * prompt_loss_len + completion_ids
    padded_logprobs = [0.0] * prompt_loss_len + old_logprobs
    padded_advantages = [0.0] * prompt_loss_len + advantages.tolist()
    if not (
        len(input_ids)
        == len(target_ids)
        == len(padded_logprobs)
        == len(padded_advantages)
    ):
        raise ValueError("OPSD Datum 字段长度不一致")

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
    row: dict[str, Any],
    args: argparse.Namespace,
    sampling_params: trio.SamplingParams,
) -> PromptRollout:
    """完成单题 Student 采样、Teacher 打分和 OPD Datum 构造。"""
    problem = str(row["problem"]).strip()
    solution = str(row["solution"]).strip()
    if not problem or not solution:
        raise ValueError("OPSD row 的 problem/solution 不能为空")

    student_prompt_ids = build_student_prompt_ids(
        tokenizer,
        problem,
        args.student_thinking,
    )
    teacher_prompt_ids = build_teacher_prompt_ids(
        tokenizer,
        problem,
        solution,
        args.teacher_thinking,
    )
    sample_result = student_sampler.sample(
        prompt=trio.ModelInput.from_ints(student_prompt_ids),
        num_samples=args.group_size,
        sampling_params=sampling_params,
        return_text=False,
    ).result()

    datums: list[trio.Datum] = []
    reverse_kls: list[float] = []
    student_logprobs: list[float] = []
    teacher_logprobs: list[float] = []
    completion_token_counts: list[int] = []
    sample_text = None

    for sequence in sample_result.sequences:
        completion_ids = list(sequence.tokens)
        if not completion_ids:
            continue
        if len(sequence.logprobs) != len(completion_ids) or any(
            value is None for value in sequence.logprobs
        ):
            raise ValueError("Student completion token/logprob 无法一一对齐")
        student_lps = [float(value) for value in sequence.logprobs]
        teacher_lps = teacher_completion_logprobs(
            teacher_client,
            teacher_prompt_ids,
            completion_ids,
        )

        # 这是 KL(pi_student || pi_teacher) 的 sampled-token Monte Carlo 项。
        reverse_kl = np.asarray(student_lps) - np.asarray(teacher_lps)
        advantages = -args.kl_penalty_coef * reverse_kl
        datums.append(
            build_opd_datum(
                student_prompt_ids,
                completion_ids,
                student_lps,
                advantages,
            )
        )
        reverse_kls.extend(reverse_kl.tolist())
        student_logprobs.extend(student_lps)
        teacher_logprobs.extend(teacher_lps)
        completion_token_counts.append(len(completion_ids))
        if sample_text is None:
            sample_text = tokenizer.decode(completion_ids, skip_special_tokens=False)

    return PromptRollout(
        datums=datums,
        reverse_kls=reverse_kls,
        student_logprobs=student_logprobs,
        teacher_logprobs=teacher_logprobs,
        completion_token_counts=completion_token_counts,
        sample_text=sample_text,
    )


def numeric_trainer_metrics(result: Any) -> dict[str, float]:
    """筛选 PyTRIO 返回的数值指标并添加 trainer 命名空间。"""
    metrics: dict[str, float] = {}
    for key, value in dict(result.metrics).items():
        try:
            metrics[f"trainer/{key}"] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


def start_swanlab(
    args: argparse.Namespace,
    dataset_size: int,
    total_steps: int,
) -> Any | None:
    """按配置创建 SwanLab 实验，并记录本次训练的静态元信息。"""
    if not args.swanlab:
        return None
    config = vars(args).copy()
    config["dataset_path"] = str(args.dataset_path)
    config["dataset_size"] = dataset_size
    config["total_steps"] = total_steps
    config["teacher"] = "fixed base model with privileged solution"
    config["objective"] = "sampled-token reverse KL via importance_sampling"
    return swanlab.init(
        project=args.swanlab_project,
        name=args.run_name,
        workspace=args.swanlab_workspace,
        mode=args.swanlab_mode,
        config=config,
        tags=["PyTRIO", "OPSD", "OpenThoughts", "fixed-teacher", "sync"],
        log_dir=str(SCRIPT_DIR / "swanlog"),
    )


def save_checkpoint(
    training_client: Any,
    swanlab_run: Any | None,
    run_name: str,
    completed_steps: int,
) -> tuple[str, str]:
    """同时保存可续训 state 和可用于评测的 sampler weights。"""
    tag = f"step{completed_steps:06d}"
    name = f"{run_name}-{tag}"

    # Train state 包含模型和优化器；sampler weights 用于推理与 AIME25 评测。
    state_result = training_client.save_state(name=name).result()
    weights_result = training_client.save_weights_for_sampler(name=name).result()
    print(f"Saved state [{tag}]: {state_result.path}")
    print(f"Saved sampler weights [{tag}]: {weights_result.path}")
    if swanlab_run is not None:
        swanlab.log(
            {
                "checkpoint/step": completed_steps,
                "checkpoint/state_path": swanlab.Text(state_result.path),
                "checkpoint/sampler_weights_path": swanlab.Text(weights_result.path),
            },
            step=completed_steps,
        )
    return state_result.path, weights_result.path


def finish_swanlab(swanlab_run: Any | None, error: BaseException | None) -> None:
    """根据正常结束、中断或异常设置 SwanLab 运行状态。"""
    if swanlab_run is None:
        return
    if error is None:
        swanlab.finish()
    elif isinstance(error, KeyboardInterrupt):
        swanlab.finish(state="aborted")
    else:
        swanlab.finish(state="crashed", error=str(error))


def train(args: argparse.Namespace) -> None:
    """执行同步 OPSD 主循环，并定期记录指标和保存双份权重。"""
    # 固定本地随机源；数据集 shuffle、远程训练和采样也会复用同一个 seed。
    random.seed(args.seed)
    np.random.seed(args.seed)
    dataset = load_training_dataset(args)

    # steps=0 表示完整遍历一次数据，否则严格执行用户指定的 step 数。
    full_dataset_run = args.steps == 0
    total_steps = (
        (len(dataset) + args.batch_size - 1) // args.batch_size
        if full_dataset_run
        else args.steps
    )
    print(f"Loaded OPSD examples: {len(dataset):,}")
    print(f"Training steps: {total_steps}")
    print(f"Run name: {args.run_name}")
    print("Teacher: fixed base model; privileged field: solution")
    print("Objective: sampled-token reverse KL (PyTRIO importance_sampling)")

    # ServiceClient 是创建远程训练客户端和采样客户端的统一入口。
    service_client = trio.ServiceClient()
    swanlab_run = None
    caught_error: BaseException | None = None
    try:
        # Student 只训练 LoRA；attention 和 MLP 默认参与训练，unembed 可选。
        training_client = service_client.create_lora_training_client(
            base_model=args.base_model,
            rank=args.lora_rank,
            seed=args.seed,
            train_attn=True,
            train_mlp=True,
            train_unembed=args.train_unembed,
        )
        # 不传 model_path，teacher 始终是 step-0 base policy，不随 student LoRA 更新。
        teacher_client = service_client.create_sampling_client(
            base_model=args.base_model
        )
        tokenizer = teacher_client.get_tokenizer()

        # 所有 Student rollout 共用这一组采样参数。
        sampling_params = trio.SamplingParams(
            max_tokens=args.max_tokens,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stop=list(
                dict.fromkeys(
                    token for token in (tokenizer.eos_token, "<|im_end|>") if token
                )
            ),
        )
        # 每个 step 的所有 Datum 聚合后，只执行一次 optimizer update。
        adam = trio.AdamParams(
            learning_rate=args.learning_rate,
            beta1=args.beta1,
            beta2=args.beta2,
        )
        swanlab_run = start_swanlab(args, len(dataset), total_steps)
        student_sampler = None
        saved_steps: set[int] = set()

        for step in range(total_steps):
            step_start = time.time()

            # 刷新 sampler 权重；间隔为 1 时，每步 rollout 都来自最新 Student。
            if student_sampler is None or step % args.sampler_refresh_steps == 0:
                student_sampler = training_client.save_weights_and_get_sampling_client()

            # 从训练开始时已经打乱的数据集中，按 step 取出当前 problem batch。
            batch = batch_for_step(
                dataset,
                step,
                args.batch_size,
                full_dataset_run,
            )

            # 同步版逐题执行：Student 采样 -> 固定 Teacher 打分 -> 构造 OPD Datum。
            rollouts = [
                run_prompt_rollout(
                    student_sampler,
                    teacher_client,
                    tokenizer,
                    row,
                    args,
                    sampling_params,
                )
                for row in tqdm(
                    batch, desc=f"OPSD sync step {step + 1}", unit="problem"
                )
            ]

            # 将题目级结果展平，组成一次远程训练更新及 SwanLab 聚合指标。
            datums = [datum for rollout in rollouts for datum in rollout.datums]
            reverse_kls = [
                value for rollout in rollouts for value in rollout.reverse_kls
            ]
            student_lps = [
                value for rollout in rollouts for value in rollout.student_logprobs
            ]
            teacher_lps = [
                value for rollout in rollouts for value in rollout.teacher_logprobs
            ]
            completion_counts = [
                value
                for rollout in rollouts
                for value in rollout.completion_token_counts
            ]
            if not datums:
                raise RuntimeError("本 step 没有生成有效 OPSD Datum")

            # 先提交 forward/backward 和 optimizer，再依次等待两个远程任务完成。
            fwd_bwd_future = training_client.forward_backward(
                datums,
                loss_fn="importance_sampling",
            )
            optim_future = training_client.optim_step(adam)
            fwd_bwd_result = fwd_bwd_future.result()
            optim_future.result()

            # 汇总当前 step 的训练、OPSD 差异和整体耗时指标。
            completed_steps = step + 1
            elapsed = time.time() - step_start
            completion_tokens_total = int(sum(completion_counts))
            metrics: dict[str, Any] = {
                "train/step": completed_steps,
                "train/learning_rate": args.learning_rate,
                "data/prompts": len(batch),
                "data/datums": len(datums),
                "data/completion_tokens_mean": float(np.mean(completion_counts)),
                "data/completion_tokens_total": completion_tokens_total,
                "opd/reverse_kl_mean": float(np.mean(reverse_kls)),
                "opd/reverse_kl_std": float(np.std(reverse_kls)),
                "opd/advantage_mean": float(
                    -args.kl_penalty_coef * np.mean(reverse_kls)
                ),
                "opd/student_logprob_mean": float(np.mean(student_lps)),
                "opd/teacher_logprob_mean": float(np.mean(teacher_lps)),
                "time/step_elapsed_time": elapsed,
            }
            metrics.update(numeric_trainer_metrics(fwd_bwd_result))

            # 按配置抽取一条 Student completion，避免每步上传大量文本。
            if (
                args.log_sample_every_steps > 0
                and completed_steps % args.log_sample_every_steps == 0
            ):
                sample_text = next(
                    (
                        rollout.sample_text
                        for rollout in rollouts
                        if rollout.sample_text
                    ),
                    None,
                )
                if sample_text is not None:
                    metrics["sample/student_completion"] = swanlab.Text(sample_text)

            # SwanLab 写入放在远程训练完成之后，确保 step 和权重更新严格对齐。
            if swanlab_run is not None:
                swanlab.log(metrics, step=completed_steps)

            print(
                f"step {completed_steps:03d}/{total_steps} | datums {len(datums)} | "
                f"avg_tokens {metrics['data/completion_tokens_mean']:.1f} | "
                f"reverse_kl {metrics['opd/reverse_kl_mean']:.4f} | "
                f"time {elapsed:.2f}s"
            )

            # 定期同时保存可续训 state 和可直接评测的 sampler weights。
            if (
                args.save_every_steps > 0
                and completed_steps % args.save_every_steps == 0
            ):
                save_checkpoint(
                    training_client,
                    swanlab_run,
                    args.run_name,
                    completed_steps,
                )
                saved_steps.add(completed_steps)

        # 如果最后一步不在定期保存点上，额外保存最终双份 checkpoint。
        if total_steps not in saved_steps:
            save_checkpoint(
                training_client,
                swanlab_run,
                args.run_name,
                total_steps,
            )
        print(f"Completed {total_steps} OPSD steps")
    except BaseException as error:
        caught_error = error
        raise
    finally:
        # 正常结束、中断或异常时都正确关闭 SwanLab run。
        finish_swanlab(swanlab_run, caught_error)


if __name__ == "__main__":
    started = time.time()
    train(parse_args())
    print(f"All done in {time.time() - started:.2f}s")
