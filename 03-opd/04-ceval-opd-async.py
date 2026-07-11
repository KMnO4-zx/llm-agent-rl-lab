"""异步版第三阶段 C-Eval OPD：4B base teacher -> Medical OPD student。

本阶段先把第二阶段 Medical OPD 的 Train state 加载到临时 TrainingClient，
保存一份 source-copy Train state。随后从 source-copy state 创建正式 TrainingClient，
并在任何 OPD update 之前，于正式 run 内保存 step000000 state + sampler weights。
训练时使用原始 Qwen3.5-4B base 作为通用能力锚点，在 C-Eval non-med
train pool 上让 student 自主采样，base teacher 对同一条 student completion
计算逐 token logprob，并通过 reverse KL advantage 更新 student。

训练只使用 C-Eval 的题目和选项，不使用 answer_idx。MedQA-ZH 和 C-Eval test
都不进入训练，只用于评测各 step checkpoint：
- MedQA-ZH @ max_tokens=1024 检查医疗能力是否保持；
- C-Eval non-med @ max_tokens=8192 检查通用能力是否恢复。

小成本试跑：
uv run python 04-ceval-opd-async.py \
    --student-state-path YOUR_MEDICAL_OPD_TRAIN_STATE_PATH \
    --steps 10 \
    --batch-size 2 \
    --group-size 2 \
    --sample-size 100 \
    --max-tokens 2048 \
    --learning-rate 5e-6 \
    --save-every-steps 5 \
    --swanlab-mode disabled

第三阶段正式试验：
uv run python 04-ceval-opd-async.py \
    --student-state-path YOUR_MEDICAL_OPD_TRAIN_STATE_PATH \
    --steps 300 \
    --batch-size 4 \
    --group-size 4 \
    --sample-size 0 \
    --max-tokens 2048 \
    --learning-rate 5e-6 \
    --save-every-steps 50 \
    --swanlab-mode online

注意：
- `--student-state-path` 必须是 `save_state` 生成的 Train state，不能传 sampler weights。
- 原始 Medical OPD state 只用于创建临时 client，不会直接进入第三阶段训练循环。
- 临时 client 只保存 source-copy state，用它创建正式 TrainingClient。
- 正式 client 会在训练前保存本 run 的 step000000 state + sampler weights。
- 两次加载都只加载模型权重，不继承第二阶段优化器状态。
- `--max-tokens 2048` 是训练 rollout 的 completion 预算，不是 C-Eval 评测预算。
"""

from __future__ import annotations

import argparse
import asyncio
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
DEFAULT_DATASET_PATH = SCRIPT_DIR / "datasets" / "ceval-non-med" / "opd_train.jsonl"
FORBIDDEN_EVAL_FILENAMES = {"test_sample.jsonl", "test_pool.jsonl", "all.jsonl"}
DEFAULT_SYSTEM_MESSAGE = (
    "你是中文单项选择题作答助手。请在内部完成必要推理，"
    "但最终回答只能包含 A、B、C、D 中的一个大写字母，"
    "不要输出推理过程、解释、标点或其他文字。"
)
LOSS_FNS = ("importance_sampling", "ppo")


@dataclass(frozen=True)
class PromptRollout:
    datums: list[trio.Datum]
    reverse_kls: list[float]
    completion_token_counts: list[int]


class ChineseArgumentParser(argparse.ArgumentParser):
    def format_usage(self) -> str:
        """把 argparse 的 usage 标题替换成中文。"""
        return super().format_usage().replace("usage:", "用法:", 1)

    def format_help(self) -> str:
        """把 argparse 帮助信息中的 usage 标题替换成中文。"""
        return super().format_help().replace("usage:", "用法:", 1)


def model_slug(base_model: str) -> str:
    """把模型名转换成适合放进运行名的短字符串。"""
    name = base_model.rsplit("/", 1)[-1].lower().replace("qwen3.5", "qwen35")
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def build_run_name(args: argparse.Namespace) -> str:
    """根据 teacher、loss 和训练步数自动生成运行名。"""
    loss_slug = args.loss_fn.replace("_", "-")
    steps_slug = "full" if args.steps == 0 else f"steps{args.steps}"
    return (
        f"opd-ceval-async-{model_slug(args.teacher_base_model)}-"
        f"base-anchor-{loss_slug}-{steps_slug}"
    )


def parse_args() -> argparse.Namespace:
    """解析并校验第三阶段 C-Eval OPD 的命令行参数。"""
    parser = ChineseArgumentParser(
        description=(
            "PyTRIO 异步版第三阶段 C-Eval OPD：base teacher -> Medical OPD student"
        ),
        add_help=False,
    )
    parser._optionals.title = "选项"
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="C-Eval non-med OPD train JSONL；不能使用 test 文件",
    )
    parser.add_argument("--sample-size", type=int, default=0, help="随机抽样 prompt 数；0 表示全用")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    parser.add_argument(
        "--student-state-path",
        required=True,
        help="Medical OPD 的 trio:// Train state 路径；不能传 sampler_weights",
    )
    parser.add_argument(
        "--teacher-base-model",
        default="Qwen/Qwen3.5-4B",
        help="作为通用能力锚点的原始 base model",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="训练 step 数；0 表示完整遍历当前数据一次",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="每个 step 使用的 C-Eval prompt 数")
    parser.add_argument("--group-size", type=int, default=4, help="每个 prompt 的 student completion 数")
    parser.add_argument("--max-tokens", type=int, default=2048, help="每条训练 completion 最多生成 token 数")
    parser.add_argument("--temperature", type=float, default=1.0, help="student rollout temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="student rollout top_p")
    parser.add_argument("--top-k", type=int, default=-1, help="student rollout top_k；-1 表示不限制")
    parser.add_argument(
        "--system-message",
        default=DEFAULT_SYSTEM_MESSAGE,
        help="与 C-Eval 评测一致的 system message；空字符串表示不加",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="chat template 是否启用 thinking",
    )
    parser.add_argument("--kl-penalty-coef", type=float, default=1.0, help="reverse KL advantage 系数")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="第三阶段 Adam 学习率")
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
        help="每多少 step 保存 state + sampler weights；0 表示只保存最终 checkpoint",
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
    if args.kl_penalty_coef <= 0:
        raise ValueError("--kl-penalty-coef must be > 0")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be > 0")
    if args.temperature < 0:
        raise ValueError("--temperature must be >= 0")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if args.save_every_steps < 0:
        raise ValueError("--save-every-steps must be >= 0")
    if "/sampler_weights/" in args.student_state_path:
        raise ValueError(
            "--student-state-path 必须传 save_state 生成的 Train state，不能传 sampler_weights"
        )

    args.run_name = build_run_name(args)
    return args


def build_ceval_prompt(row: dict[str, Any], line_number: int) -> str:
    """把一条 C-Eval 题目和四个选项整理成统一训练 prompt。"""
    question = str(row.get("question", "")).strip()
    options = row.get("options")
    if not isinstance(options, dict):
        raise ValueError(f"C-Eval 第 {line_number} 行的 options 不是对象")
    normalized_options = {
        key: str(options.get(key, "")).strip() for key in ("A", "B", "C", "D")
    }
    if not question or not all(normalized_options.values()):
        raise ValueError(f"C-Eval 第 {line_number} 行缺少 question 或 A/B/C/D 选项")
    return "\n".join(
        [
            "以下是中国考试中的单项选择题。请仔细思考，并只输出最终答案选项字母。",
            "",
            f"题目：{question}",
            f"A. {normalized_options['A']}",
            f"B. {normalized_options['B']}",
            f"C. {normalized_options['C']}",
            f"D. {normalized_options['D']}",
            "",
            "答案：",
        ]
    )


def load_ceval_prompts(args: argparse.Namespace) -> list[dict[str, str]]:
    """加载、校验并打乱 C-Eval OPD 训练 prompt。"""
    if not args.dataset_path.exists():
        raise FileNotFoundError(
            f"找不到 C-Eval OPD 数据：{args.dataset_path}，请先运行 00-download-dataset.py"
        )
    if (
        args.dataset_path.parent.name == "ceval-non-med"
        and args.dataset_path.name in FORBIDDEN_EVAL_FILENAMES
    ):
        raise ValueError(
            f"禁止使用 C-Eval 测试数据训练：{args.dataset_path.name}；请使用 opd_train.jsonl"
        )

    rows: list[dict[str, str]] = []
    with args.dataset_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            raw_row = json.loads(line)
            rows.append(
                {
                    "row_id": str(raw_row.get("row_id", line_number)),
                    "subject": str(raw_row.get("subject", "unknown")),
                    "prompt": build_ceval_prompt(raw_row, line_number),
                }
            )

    if not rows:
        raise ValueError(f"C-Eval OPD 数据为空：{args.dataset_path}")
    random.Random(args.seed).shuffle(rows)
    if args.sample_size > 0:
        rows = rows[: min(args.sample_size, len(rows))]
    return rows


def build_prompt_tokens(
    tokenizer: Any,
    prompt: str,
    system_message: str,
    enable_thinking: bool,
) -> list[int]:
    """应用 chat template，把 system + C-Eval prompt 转成 token。"""
    messages = []
    if system_message.strip():
        messages.append({"role": "system", "content": system_message.strip()})
    messages.append({"role": "user", "content": prompt})
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("Prompt tokens are empty")
    return prompt_ids


async def completion_teacher_logprobs_async(
    teacher_client: Any,
    prompt_ids: list[int],
    completion_ids: list[int],
) -> list[float]:
    """计算 base teacher 对 student completion 的逐 token logprob。"""
    all_ids = prompt_ids + completion_ids
    all_logprobs = await teacher_client.compute_logprobs_async(
        trio.ModelInput.from_ints(all_ids)
    )
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
    """把一条 student 轨迹及其 advantage 组装成 PyTRIO OPD Datum。"""
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


async def run_prompt_rollout_async(
    student_sampler: Any,
    teacher_client: Any,
    tokenizer: Any,
    row: dict[str, str],
    args: argparse.Namespace,
    sampling_params: trio.SamplingParams,
) -> PromptRollout:
    """完成一个 prompt 的 student 采样、teacher 打分和 OPD Datum 构造。"""
    prompt_ids = build_prompt_tokens(
        tokenizer=tokenizer,
        prompt=row["prompt"],
        system_message=args.system_message,
        enable_thinking=args.enable_thinking,
    )
    sample_result = await student_sampler.sample_async(
        prompt=trio.ModelInput.from_ints(prompt_ids),
        num_samples=args.group_size,
        sampling_params=sampling_params,
        return_text=False,
    )
    sequences = [sequence for sequence in sample_result.sequences if sequence.tokens]
    if not sequences:
        raise RuntimeError(f"Student 没有为 C-Eval prompt 生成有效 completion：{row['row_id']}")

    teacher_logprobs_list = await asyncio.gather(
        *(
            completion_teacher_logprobs_async(
                teacher_client=teacher_client,
                prompt_ids=prompt_ids,
                completion_ids=list(sequence.tokens),
            )
            for sequence in sequences
        )
    )

    datums: list[trio.Datum] = []
    reverse_kls: list[float] = []
    completion_token_counts: list[int] = []
    for sequence, teacher_logprobs in zip(
        sequences,
        teacher_logprobs_list,
        strict=True,
    ):
        completion_ids = list(sequence.tokens)
        student_logprobs = [float(value) for value in sequence.logprobs]
        if len(student_logprobs) != len(completion_ids):
            raise ValueError("Student token/logprob length mismatch")
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
    """按当前 step 选择训练 batch，完整遍历时不重复最后一批。"""
    if full_dataset_run:
        start = step * batch_size
        return rows[start : start + batch_size]
    return [rows[(step * batch_size + offset) % len(rows)] for offset in range(batch_size)]


def pick_loss_metric(metrics: dict[str, float]) -> float | None:
    """从 PyTRIO 返回的不同指标命名中提取 loss。"""
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
    """按命令行配置初始化 SwanLab 运行记录。"""
    if not args.swanlab:
        return None
    config = vars(args).copy()
    config["dataset_path"] = str(args.dataset_path)
    config["dataset_size"] = dataset_size
    config["total_steps"] = total_steps
    config["optimizer_state_restored"] = False
    return swanlab.init(
        project=args.swanlab_project,
        name=args.run_name,
        workspace=args.swanlab_workspace,
        mode=args.swanlab_mode,
        config=config,
        tags=["PyTrio", "OPD", "C-Eval", "base-anchor", "async"],
        log_dir=str(SCRIPT_DIR / "swanlog"),
    )


async def save_checkpoint_async(
    training_client: Any,
    swanlab_run: Any | None,
    args: argparse.Namespace,
    completed_steps: int,
) -> tuple[str, str]:
    """同时保存可续训 state 和可评测 sampler weights。"""
    tag = f"step{completed_steps:06d}"
    name = f"{args.run_name}-{tag}"
    state_future = await training_client.save_state_async(name=name)
    state_result = await state_future
    weights_future = await training_client.save_weights_for_sampler_async(name=name)
    weights_result = await weights_future
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


async def save_source_state_copy_async(
    training_client: Any,
    swanlab_run: Any | None,
    args: argparse.Namespace,
) -> str:
    """把原始 Medical OPD 权重保存成供正式 TrainingClient 加载的 state 副本。"""
    name = f"{args.run_name}-source-copy"
    state_future = await training_client.save_state_async(name=name)
    state_result = await state_future
    print(f"Saved source-copy state: {state_result.path}")
    if swanlab_run is not None:
        swanlab.log(
            {"save/source_copy_state_path": swanlab.Text(state_result.path)},
            step=0,
        )
    return state_result.path


async def train(args: argparse.Namespace) -> None:
    """克隆 Medical OPD state，并执行第三阶段异步 C-Eval OPD 训练。"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    rows = load_ceval_prompts(args)
    full_dataset_run = args.steps == 0
    total_steps = (
        (len(rows) + args.batch_size - 1) // args.batch_size
        if full_dataset_run
        else args.steps
    )
    print(f"Loaded C-Eval non-med OPD prompts: {len(rows)}")
    print(f"Training steps: {total_steps}")
    print(f"Run name: {args.run_name}")
    print(f"Student Train state: {args.student_state_path}")
    print(f"Base-anchor teacher: {args.teacher_base_model}")
    print("The source state will be cloned before any C-Eval OPD update.")
    print("answer_idx is ignored; C-Eval test data are not loaded by this training script.")

    service_client = trio.ServiceClient()
    swanlab_run = None
    try:
        swanlab_run = start_swanlab(args, len(rows), total_steps)

        print("Loading Medical OPD weights into a temporary TrainingClient...")
        source_client = await service_client.create_training_client_from_state_async(
            path=args.student_state_path
        )
        source_copy_state_path = await save_source_state_copy_async(
            training_client=source_client,
            swanlab_run=swanlab_run,
            args=args,
        )
        source_client = None

        print("Creating the stage-3 TrainingClient from the source-copy state...")
        training_client = await service_client.create_training_client_from_state_async(
            path=source_copy_state_path
        )
        initial_state_path, initial_weights_path = await save_checkpoint_async(
            training_client=training_client,
            swanlab_run=swanlab_run,
            args=args,
            completed_steps=0,
        )
        print(f"Saved formal-run step000000 state: {initial_state_path}")
        print(f"Saved formal-run step000000 sampler weights: {initial_weights_path}")
        teacher_client = await service_client.create_sampling_client_async(
            base_model=args.teacher_base_model
        )
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
        student_sampler = None

        for step in range(total_steps):
            step_start = time.time()
            if student_sampler is None or step % args.sampler_refresh_steps == 0:
                student_sampler = (
                    await training_client.save_weights_and_get_sampling_client_async()
                )

            batch = batch_for_step(
                rows=rows,
                step=step,
                batch_size=args.batch_size,
                full_dataset_run=full_dataset_run,
            )
            with tqdm(
                total=len(batch),
                desc=f"C-Eval OPD async step {step + 1}",
                unit="prompt",
            ) as progress_bar:

                async def run_and_track(row: dict[str, str]) -> PromptRollout:
                    """执行单题 rollout，并同步更新当前 step 的进度条。"""
                    rollout = await run_prompt_rollout_async(
                        student_sampler=student_sampler,
                        teacher_client=teacher_client,
                        tokenizer=tokenizer,
                        row=row,
                        args=args,
                        sampling_params=sampling_params,
                    )
                    progress_bar.update(1)
                    return rollout

                rollouts = await asyncio.gather(
                    *(run_and_track(row) for row in batch)
                )

            datums = [datum for rollout in rollouts for datum in rollout.datums]
            reverse_kls = [value for rollout in rollouts for value in rollout.reverse_kls]
            completion_counts = [
                value
                for rollout in rollouts
                for value in rollout.completion_token_counts
            ]
            if not datums:
                raise RuntimeError("No C-Eval OPD datums were built")

            fwd_bwd_future = await training_client.forward_backward_async(
                datums,
                loss_fn=args.loss_fn,
            )
            optim_future = await training_client.optim_step_async(adam)
            fwd_bwd_result = await fwd_bwd_future
            await optim_future

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
            tqdm.write(
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
                await save_checkpoint_async(
                    training_client=training_client,
                    swanlab_run=swanlab_run,
                    args=args,
                    completed_steps=completed_steps,
                )
    finally:
        if swanlab_run is not None:
            swanlab.finish()


def main() -> None:
    """启动异步训练并打印总耗时。"""
    started_at = time.time()
    asyncio.run(train(parse_args()))
    print("#" * 50)
    print("# all done")
    print(f"# train cost {time.time() - started_at:.2f}s")
    print("#" * 50)


if __name__ == "__main__":
    main()
