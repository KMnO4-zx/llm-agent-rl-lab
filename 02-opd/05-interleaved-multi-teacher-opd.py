"""Interleaved Multi-Teacher OPD：按 step 交替蒸馏医疗与通用能力。

Student 始终从 Qwen3.5-4B base 创建 fresh LoRA。Medical step 使用第一阶段
Medical SFT sampler weights 作为 Teacher；C-Eval step 使用原始 4B base model
作为 Teacher。每个 step 都由当前 Student 自主采样，再由对应 Teacher 对同一条
Student completion 计算逐 token logprob，并通过 reverse KL advantage 更新 Student。

默认采用 1:1 调度：Medical -> C-Eval，600 step 对应两类任务各 300 step。
将 ``--medical-steps-per-cycle`` 设为 2 即得到 2:1 调度：
Medical -> Medical -> C-Eval。

小成本试跑：
uv run python 05-interleaved-multi-teacher-opd.py \
    --medical-teacher-model-path YOUR_SFT_SAMPLER_WEIGHTS_PATH \
    --steps 10 \
    --medical-sample-size 100 \
    --ceval-sample-size 100 \
    --batch-size 2 \
    --group-size 2 \
    --save-every-steps 10 \
    --swanlab-mode disabled

1:1 正式实验：
uv run python 05-interleaved-multi-teacher-opd.py \
    --medical-teacher-model-path YOUR_SFT_SAMPLER_WEIGHTS_PATH \
    --steps 600 \
    --medical-steps-per-cycle 1 \
    --ceval-steps-per-cycle 1 \
    --batch-size 4 \
    --group-size 4 \
    --max-tokens 2048 \
    --learning-rate 1e-5 \
    --save-every-steps 50 \
    --swanlab-mode online

2:1 正式实验：
uv run python 05-interleaved-multi-teacher-opd.py \
    --medical-teacher-model-path YOUR_SFT_SAMPLER_WEIGHTS_PATH \
    --steps 600 \
    --medical-steps-per-cycle 2 \
    --ceval-steps-per-cycle 1 \
    --batch-size 4 \
    --group-size 4 \
    --max-tokens 2048 \
    --learning-rate 1e-5 \
    --save-every-steps 60 \
    --swanlab-mode online

注意：
- Medical 与 C-Eval 都只使用 prompt，不使用数据集中的参考答案。
- 两个 Teacher 只对 Student 实际生成的 completion 打分，不自行生成训练答案。
- 两类任务共享一个 TrainingClient、一个 LoRA Student 和同一份 Adam 状态。
- 正式训练前会在同一个 run 内保存 step000000 state 与 sampler weights。
- checkpoint 最好保存在完整调度周期结束后：1:1 用 2 的倍数，2:1 用 3 的倍数。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytrio as trio
import swanlab
from tqdm import tqdm


trio.configure(timeout=600)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MEDICAL_DATASET_PATH = (
    SCRIPT_DIR / "datasets" / "medical-o1-reasoning-SFT-zh" / "train.jsonl"
)
DEFAULT_CEVAL_DATASET_PATH = (
    SCRIPT_DIR / "datasets" / "ceval-non-med" / "opd_train.jsonl"
)
FORBIDDEN_CEVAL_FILENAMES = {"test_sample.jsonl", "test_pool.jsonl", "all.jsonl"}
DEFAULT_MEDICAL_SYSTEM_MESSAGE = (
    "你是一个中文医疗问答助手。请根据题目给出严谨的医学推理和最终答案。"
)
DEFAULT_CEVAL_SYSTEM_MESSAGE = (
    "你是中文单项选择题作答助手。请先完成必要推理，"
    "最终回答只能包含 A、B、C、D 中的一个大写字母，"
    "不要在最终答案中加入解释、标点或其他文字。"
)
LOSS_FNS = ("importance_sampling", "ppo")
TaskName = Literal["medical", "ceval"]


@dataclass(frozen=True)
class PromptRollout:
    """保存一个 prompt 产生的 OPD datums 和统计量。"""

    datums: list[trio.Datum]
    reverse_kls: list[float]
    completion_token_counts: list[int]


class ChineseArgumentParser(argparse.ArgumentParser):
    """把 argparse 的 usage 标题替换为中文。"""

    def format_usage(self) -> str:
        """生成中文 usage 标题。"""

        return super().format_usage().replace("usage:", "用法:", 1)

    def format_help(self) -> str:
        """生成中文 help 中的 usage 标题。"""

        return super().format_help().replace("usage:", "用法:", 1)


def model_slug(base_model: str) -> str:
    """把模型名转换为适合放进 run name 的短字符串。"""

    name = base_model.rsplit("/", 1)[-1].lower().replace("qwen3.5", "qwen35")
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def build_run_name(args: argparse.Namespace) -> str:
    """根据模型、任务比例、loss 和 step 数生成运行名。"""

    loss_slug = args.loss_fn.replace("_", "-")
    steps_slug = "full" if args.steps == 0 else f"steps{args.steps}"
    ratio_slug = (
        f"m{args.medical_steps_per_cycle}-c{args.ceval_steps_per_cycle}"
    )
    return (
        f"opd-interleaved-multi-teacher-{model_slug(args.base_model)}-"
        f"{ratio_slug}-{loss_slug}-{steps_slug}"
    )


def parse_args() -> argparse.Namespace:
    """解析并校验 Interleaved Multi-Teacher OPD 参数。"""

    parser = ChineseArgumentParser(
        description=(
            "PyTRIO Interleaved Multi-Teacher OPD：Medical SFT teacher 与 "
            "base teacher 按 step 交替蒸馏 fresh base student"
        ),
        add_help=False,
    )
    parser._optionals.title = "选项"
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出")

    parser.add_argument(
        "--medical-dataset-path",
        type=Path,
        default=DEFAULT_MEDICAL_DATASET_PATH,
        help="medical-o1-reasoning-SFT-zh train JSONL",
    )
    parser.add_argument(
        "--ceval-dataset-path",
        type=Path,
        default=DEFAULT_CEVAL_DATASET_PATH,
        help="C-Eval non-med OPD train JSONL；不能使用 test 文件",
    )
    parser.add_argument(
        "--medical-sample-size",
        type=int,
        default=0,
        help="随机抽样 medical prompt 数；0 表示全用",
    )
    parser.add_argument(
        "--ceval-sample-size",
        type=int,
        default=0,
        help="随机抽样 C-Eval prompt 数；0 表示全用",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3.5-4B",
        help="fresh LoRA student 与 C-Eval base teacher 的基础模型",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="fresh student 的 LoRA rank，范围 4-64",
    )
    parser.add_argument(
        "--medical-teacher-base-model",
        default=None,
        help="Medical SFT teacher 的基础模型；默认和 student base 相同",
    )
    parser.add_argument(
        "--medical-teacher-model-path",
        required=True,
        help="Medical SFT 的 trio:// sampler_weights 路径",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=600,
        help="总 optimizer step 数；0 表示运行完整周期直到两类数据都至少遍历一次",
    )
    parser.add_argument(
        "--medical-steps-per-cycle",
        type=int,
        default=1,
        help="每个调度周期包含的 Medical step 数",
    )
    parser.add_argument(
        "--ceval-steps-per-cycle",
        type=int,
        default=1,
        help="每个调度周期包含的 C-Eval step 数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="每个 step 使用的当前任务 prompt 数",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="每个 prompt 的 Student completion 数",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="每条 Student completion 最多生成 token 数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Student rollout temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Student rollout top_p",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Student rollout top_k；-1 表示不限制",
    )
    parser.add_argument(
        "--medical-system-message",
        default=DEFAULT_MEDICAL_SYSTEM_MESSAGE,
        help="Medical step 使用的 system message；空字符串表示不加",
    )
    parser.add_argument(
        "--ceval-system-message",
        default=DEFAULT_CEVAL_SYSTEM_MESSAGE,
        help="C-Eval step 使用的 system message；空字符串表示不加",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="两类任务的 chat template 是否启用 thinking",
    )

    parser.add_argument(
        "--kl-penalty-coef",
        type=float,
        default=1.0,
        help="两类任务共用的 reverse KL advantage 系数",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="两类任务共用的 Adam 学习率",
    )
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2")
    parser.add_argument(
        "--sampler-refresh-steps",
        type=int,
        default=1,
        help="每隔多少全局 step 用最新 Student 权重刷新 sampler",
    )
    parser.add_argument(
        "--loss-fn",
        choices=LOSS_FNS,
        default="ppo",
        help="PyTRIO 内置 RL loss",
    )
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=50,
        help="每多少全局 step 保存 state + sampler weights；0 表示只保存最终 checkpoint",
    )

    parser.add_argument(
        "--swanlab",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用 SwanLab",
    )
    parser.add_argument(
        "--swanlab-project",
        default="llm-agent-rl-lab",
        help="SwanLab 项目名",
    )
    parser.add_argument("--swanlab-workspace", default=None, help="SwanLab workspace")
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "offline", "disabled"],
        default=None,
        help="SwanLab 运行模式",
    )
    args = parser.parse_args()

    positive_int_names = (
        "medical_steps_per_cycle",
        "ceval_steps_per_cycle",
        "batch_size",
        "group_size",
        "max_tokens",
        "sampler_refresh_steps",
    )
    for name in positive_int_names:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.steps < 0:
        raise ValueError("--steps must be >= 0")
    if args.medical_sample_size < 0 or args.ceval_sample_size < 0:
        raise ValueError("sample size must be >= 0")
    if not 4 <= args.lora_rank <= 64:
        raise ValueError("--lora-rank must be between 4 and 64")
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

    args.medical_teacher_base_model = (
        args.medical_teacher_base_model or args.base_model
    )
    args.run_name = build_run_name(args)
    return args


def build_ceval_prompt(row: dict[str, Any], line_number: int) -> str:
    """把 C-Eval 题目与 A/B/C/D 选项整理为训练 prompt。"""

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


def load_medical_prompts(args: argparse.Namespace) -> list[dict[str, str]]:
    """加载、校验并打乱 Medical OPD prompt。"""

    path = args.medical_dataset_path
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 medical-o1 数据：{path}，请先运行 00-download-dataset.py"
        )

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            question = str(row.get("question", "")).strip()
            if not question:
                raise ValueError(f"medical-o1 第 {line_number} 行缺少 question")
            rows.append(
                {
                    "row_id": f"medical-{line_number}",
                    "prompt": question,
                }
            )

    if not rows:
        raise ValueError(f"medical-o1 数据为空：{path}")
    random.Random(args.seed).shuffle(rows)
    if args.medical_sample_size > 0:
        rows = rows[: min(args.medical_sample_size, len(rows))]
    return rows


def load_ceval_prompts(args: argparse.Namespace) -> list[dict[str, str]]:
    """加载、校验并打乱 C-Eval OPD 训练 prompt。"""

    path = args.ceval_dataset_path
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 C-Eval OPD 数据：{path}，请先运行 00-download-dataset.py"
        )
    if (
        path.parent.name == "ceval-non-med"
        and path.name in FORBIDDEN_CEVAL_FILENAMES
    ):
        raise ValueError(
            f"禁止使用 C-Eval 测试数据训练：{path.name}；请使用 opd_train.jsonl"
        )

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                {
                    "row_id": f"ceval-{row.get('row_id', line_number)}",
                    "prompt": build_ceval_prompt(row, line_number),
                }
            )

    if not rows:
        raise ValueError(f"C-Eval OPD 数据为空：{path}")
    random.Random(args.seed + 1).shuffle(rows)
    if args.ceval_sample_size > 0:
        rows = rows[: min(args.ceval_sample_size, len(rows))]
    return rows


def build_task_schedule(args: argparse.Namespace) -> tuple[TaskName, ...]:
    """按设定比例构造固定的 Medical/C-Eval step 周期。"""

    return (
        ("medical",) * args.medical_steps_per_cycle
        + ("ceval",) * args.ceval_steps_per_cycle
    )


def resolve_total_steps(
    args: argparse.Namespace,
    medical_size: int,
    ceval_size: int,
) -> int:
    """解析总 step；steps=0 时保证两类数据都至少遍历一次。"""

    if args.steps > 0:
        return args.steps

    medical_steps = math.ceil(medical_size / args.batch_size)
    ceval_steps = math.ceil(ceval_size / args.batch_size)
    cycles = max(
        math.ceil(medical_steps / args.medical_steps_per_cycle),
        math.ceil(ceval_steps / args.ceval_steps_per_cycle),
    )
    return cycles * (
        args.medical_steps_per_cycle + args.ceval_steps_per_cycle
    )


def count_scheduled_steps(
    schedule: tuple[TaskName, ...],
    total_steps: int,
) -> dict[TaskName, int]:
    """统计给定总 step 下两类任务各自会执行多少次。"""

    counts: dict[TaskName, int] = {"medical": 0, "ceval": 0}
    for step in range(total_steps):
        counts[schedule[step % len(schedule)]] += 1
    return counts


def batch_for_task_step(
    rows: list[dict[str, str]],
    task_step: int,
    batch_size: int,
) -> list[dict[str, str]]:
    """使用任务自己的 step 下标循环取 batch，避免两类数据互相跳步。"""

    start = task_step * batch_size
    return [rows[(start + offset) % len(rows)] for offset in range(batch_size)]


def build_prompt_tokens(
    tokenizer: Any,
    prompt: str,
    system_message: str,
    enable_thinking: bool,
) -> list[int]:
    """应用当前任务的 system message 和 chat template。"""

    messages = []
    if system_message.strip():
        messages.append({"role": "system", "content": system_message.strip()})
    messages.append({"role": "user", "content": prompt.strip()})
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
    """让当前任务 Teacher 对 Student completion 计算逐 token logprob。"""

    all_logprobs = await teacher_client.compute_logprobs_async(
        trio.ModelInput.from_ints(prompt_ids + completion_ids)
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
    """把 Student 轨迹、old logprobs 与 advantage 组装成 OPD Datum。"""

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
    system_message: str,
    args: argparse.Namespace,
    sampling_params: trio.SamplingParams,
) -> PromptRollout:
    """执行一条 Student rollout，并由当前任务 Teacher 构造 OPD 信号。"""

    prompt_ids = build_prompt_tokens(
        tokenizer=tokenizer,
        prompt=row["prompt"],
        system_message=system_message,
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
        raise RuntimeError(f"Student 没有生成有效 completion：{row['row_id']}")

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


def pick_loss_metric(metrics: dict[str, float]) -> float | None:
    """从 PyTRIO 可能返回的不同指标名中提取 loss。"""

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
    medical_size: int,
    ceval_size: int,
    total_steps: int,
    scheduled_counts: dict[TaskName, int],
) -> Any | None:
    """初始化包含双数据集与任务比例信息的 SwanLab run。"""

    if not args.swanlab:
        return None
    config = vars(args).copy()
    config["medical_dataset_path"] = str(args.medical_dataset_path)
    config["ceval_dataset_path"] = str(args.ceval_dataset_path)
    config["medical_dataset_size"] = medical_size
    config["ceval_dataset_size"] = ceval_size
    config["total_steps"] = total_steps
    config["scheduled_medical_steps"] = scheduled_counts["medical"]
    config["scheduled_ceval_steps"] = scheduled_counts["ceval"]
    return swanlab.init(
        project=args.swanlab_project,
        name=args.run_name,
        workspace=args.swanlab_workspace,
        mode=args.swanlab_mode,
        config=config,
        tags=[
            "PyTrio",
            "OPD",
            "interleaved",
            "multi-teacher",
            "medical",
            "ceval",
        ],
        log_dir=str(SCRIPT_DIR / "swanlog"),
    )


async def save_checkpoint_async(
    training_client: Any,
    swanlab_run: Any | None,
    args: argparse.Namespace,
    completed_steps: int,
) -> tuple[str, str]:
    """保存可续训 state 与可评测 sampler weights。"""

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


async def train(args: argparse.Namespace) -> None:
    """运行按 step 路由数据与 Teacher 的异步 Multi-Teacher OPD。"""

    random.seed(args.seed)
    np.random.seed(args.seed)
    medical_rows = load_medical_prompts(args)
    ceval_rows = load_ceval_prompts(args)
    schedule = build_task_schedule(args)
    cycle_length = len(schedule)
    total_steps = resolve_total_steps(
        args=args,
        medical_size=len(medical_rows),
        ceval_size=len(ceval_rows),
    )
    scheduled_counts = count_scheduled_steps(schedule, total_steps)

    print(f"Loaded Medical prompts: {len(medical_rows)}")
    print(f"Loaded C-Eval non-med prompts: {len(ceval_rows)}")
    print(f"Task schedule: {' -> '.join(schedule)}")
    print(
        f"Training steps: {total_steps} "
        f"(Medical {scheduled_counts['medical']}, C-Eval {scheduled_counts['ceval']})"
    )
    print(f"Run name: {args.run_name}")
    print(f"Fresh Student base: {args.base_model}")
    print(f"Medical SFT Teacher: {args.medical_teacher_model_path}")
    print(f"C-Eval base Teacher: {args.base_model}")
    print("Medical answers and C-Eval answer_idx are not loaded by this script.")
    if args.save_every_steps > 0 and args.save_every_steps % cycle_length != 0:
        print(
            "Warning: --save-every-steps is not a multiple of the task cycle "
            f"length {cycle_length}; some checkpoints may end mid-cycle."
        )
    if total_steps % cycle_length != 0:
        print(
            f"Warning: final step {total_steps} does not complete the "
            f"{cycle_length}-step task cycle."
        )

    service_client = trio.ServiceClient()
    swanlab_run = None
    try:
        swanlab_run = start_swanlab(
            args=args,
            medical_size=len(medical_rows),
            ceval_size=len(ceval_rows),
            total_steps=total_steps,
            scheduled_counts=scheduled_counts,
        )

        print(f"Creating fresh LoRA Student from base: {args.base_model}")
        training_client = await service_client.create_lora_training_client_async(
            base_model=args.base_model,
            rank=args.lora_rank,
            seed=args.seed,
        )
        initial_state_path, initial_weights_path = await save_checkpoint_async(
            training_client=training_client,
            swanlab_run=swanlab_run,
            args=args,
            completed_steps=0,
        )
        print(f"Saved step000000 state: {initial_state_path}")
        print(f"Saved step000000 sampler weights: {initial_weights_path}")

        medical_teacher = await service_client.create_sampling_client_async(
            base_model=args.medical_teacher_base_model,
            model_path=args.medical_teacher_model_path,
        )
        base_teacher = await service_client.create_sampling_client_async(
            base_model=args.base_model
        )
        tokenizer = base_teacher.get_tokenizer()
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
        task_steps: dict[TaskName, int] = {"medical": 0, "ceval": 0}

        for step in range(total_steps):
            step_start = time.time()
            task = schedule[step % cycle_length]
            if task == "medical":
                rows = medical_rows
                teacher_client = medical_teacher
                system_message = args.medical_system_message
                task_label = "Medical"
            else:
                rows = ceval_rows
                teacher_client = base_teacher
                system_message = args.ceval_system_message
                task_label = "C-Eval"

            if student_sampler is None or step % args.sampler_refresh_steps == 0:
                student_sampler = (
                    await training_client.save_weights_and_get_sampling_client_async()
                )

            batch = batch_for_task_step(
                rows=rows,
                task_step=task_steps[task],
                batch_size=args.batch_size,
            )
            with tqdm(
                total=len(batch),
                desc=f"{task_label} OPD step {step + 1}",
                unit="prompt",
            ) as progress_bar:

                async def run_and_track(row: dict[str, str]) -> PromptRollout:
                    """执行单条 rollout，并更新当前 task step 的进度条。"""

                    rollout = await run_prompt_rollout_async(
                        student_sampler=student_sampler,
                        teacher_client=teacher_client,
                        tokenizer=tokenizer,
                        row=row,
                        system_message=system_message,
                        args=args,
                        sampling_params=sampling_params,
                    )
                    progress_bar.update(1)
                    return rollout

                rollouts = await asyncio.gather(
                    *(run_and_track(row) for row in batch)
                )

            datums = [datum for rollout in rollouts for datum in rollout.datums]
            reverse_kls = [
                value for rollout in rollouts for value in rollout.reverse_kls
            ]
            completion_counts = [
                value
                for rollout in rollouts
                for value in rollout.completion_token_counts
            ]
            if not datums:
                raise RuntimeError(f"No {task_label} OPD datums were built")

            fwd_bwd_future = await training_client.forward_backward_async(
                datums,
                loss_fn=args.loss_fn,
            )
            optim_future = await training_client.optim_step_async(adam)
            fwd_bwd_result = await fwd_bwd_future
            await optim_future

            task_steps[task] += 1
            completed_steps = step + 1
            elapsed = time.time() - step_start
            completion_tokens_total = int(sum(completion_counts))
            raw_trainer_metrics = {
                str(key): float(value)
                for key, value in dict(fwd_bwd_result.metrics).items()
            }
            loss_value = pick_loss_metric(raw_trainer_metrics)
            metrics: dict[str, float | int] = {
                f"{task}/prompts": len(batch),
                f"{task}/datums": len(datums),
                f"{task}/completion_tokens_mean": float(
                    np.mean(completion_counts)
                ),
                f"{task}/completion_tokens_total": completion_tokens_total,
                f"{task}/reverse_kl_mean": float(np.mean(reverse_kls)),
                f"{task}/reverse_kl_std": float(np.std(reverse_kls)),
                f"{task}/completion_tokens_per_second": (
                    completion_tokens_total / elapsed
                ),
                "train/step": completed_steps,
                "train/is_medical": int(task == "medical"),
                "train/medical_steps": task_steps["medical"],
                "train/ceval_steps": task_steps["ceval"],
                "train/medical_ratio": task_steps["medical"] / completed_steps,
                "train/ceval_ratio": task_steps["ceval"] / completed_steps,
                "train/learning_rate": args.learning_rate,
                "time/step_elapsed_time": elapsed,
            }
            metrics.update(
                {
                    f"{task}/trainer/{key.removeprefix('trainer/')}": value
                    for key, value in raw_trainer_metrics.items()
                }
            )
            if loss_value is not None:
                metrics[f"{task}/loss"] = loss_value
            if swanlab_run is not None:
                swanlab.log(metrics, step=completed_steps)

            loss_text = "n/a" if loss_value is None else f"{loss_value:.4f}"
            tqdm.write(
                f"step {completed_steps:03d}/{total_steps} | task {task} | "
                f"task steps M{task_steps['medical']}/C{task_steps['ceval']} | "
                f"datums {len(datums)} | "
                f"completion tokens mean "
                f"{metrics[f'{task}/completion_tokens_mean']:.1f} | "
                f"reverse_kl {metrics[f'{task}/reverse_kl_mean']:.4f} | "
                f"tokens/s "
                f"{metrics[f'{task}/completion_tokens_per_second']:.1f} | "
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

        print(
            f"Completed task steps: Medical {task_steps['medical']}, "
            f"C-Eval {task_steps['ceval']}"
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
