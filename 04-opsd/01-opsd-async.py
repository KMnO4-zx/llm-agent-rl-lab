"""PyTRIO 异步版 OPSD。

同一 step 内并发执行多个 problem 的 Student rollout 和 Teacher 打分；
optimizer 完成后再进入下一 step，保持 on-policy。

测试:
uv run python 01-opsd-async.py \
    --steps 10 \
    --batch-size 4 \
    --group-size 1 \
    --max-tokens 1024 \
    --sample-size 100 \
    --save-every-steps 1 \
    --swanlab-mode disabled

正式：
uv run python 01-opsd-async.py \
    --steps 100 \
    --batch-size 32 \
    --group-size 1 \
    --max-tokens 1024 \
    --sample-size 0 \
    --save-every-steps 25 \
    --max-concurrency 32 \
    --swanlab-mode online
"""

from __future__ import annotations

import argparse
import asyncio
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
    # 只保留仓库名后的模型名，并统一成便于搜索的短横线格式。
    name = base_model.rsplit("/", 1)[-1].lower().replace("qwen3.5", "qwen35")
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def default_run_name(args: argparse.Namespace) -> str:
    """根据模型、目标函数和步数生成默认异步实验名称。"""
    # steps=0 代表遍历全量数据，用 full 避免实验名出现歧义。
    steps = "full" if args.steps == 0 else f"steps{args.steps}"
    return f"opsd-async-{model_slug(args.base_model)}-sampled-token-{steps}"


def parse_args() -> argparse.Namespace:
    """解析并校验异步 OPSD 的训练、并发、保存和日志参数。"""
    parser = argparse.ArgumentParser(description=__doc__)

    # 数据集与随机性参数决定训练样本顺序和可复现性。
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

    # Student LoRA、rollout 和优化器参数。
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
    parser.add_argument(
        "--max-concurrency", type=int, default=8, help="远程采样/logprob 最大并发数"
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

    # SwanLab 可独立关闭，不影响远程训练和 checkpoint 保存。
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

    # 尽早拒绝非法参数，避免远程任务创建后才失败并产生额外开销。
    for name in (
        "batch_size",
        "group_size",
        "max_concurrency",
        "max_tokens",
        "sampler_refresh_steps",
    ):
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

    # 用户未指定名称时，根据关键实验配置生成稳定的默认名称。
    args.run_name = args.run_name or default_run_name(args)
    return args


def load_training_dataset(args: argparse.Namespace) -> Dataset:
    """读取本地 OPSD 数据，校验字段和行数后打乱或抽样。"""
    # 训练只读取 00-datasets.py 落盘后的本地数据，不在训练时临时下载。
    if not args.dataset_path.exists():
        raise FileNotFoundError(
            f"找不到 OPSD 数据：{args.dataset_path}\n"
            "请先运行：uv run python 04-opsd/00-datasets.py --only opsd"
        )
    loaded = load_from_disk(str(args.dataset_path))
    # 兼容 load_from_disk 返回 Dataset 或包含 train split 的 DatasetDict。
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

    # 先按 seed 打乱，再截取 sample_size，保证相同 seed 得到相同训练子集。
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
    # 数据集在加载阶段已经打乱，这里只根据 step 计算确定性的起始位置。
    start = step * batch_size
    if full_dataset_run:
        # 全量遍历模式的最后一个 batch 可以小于 batch_size。
        indices = list(range(start, min(start + batch_size, len(dataset))))
    else:
        # 固定步数模式超过一轮数据后，从数据集开头循环继续取样。
        indices = [(start + offset) % len(dataset) for offset in range(batch_size)]
    return dataset.select(indices)


def render_chat_prompt(
    tokenizer: Any,
    user_message: str,
    enable_thinking: bool,
) -> list[int]:
    """应用 chat template 并把用户消息编码为 prompt token。"""
    # Student 和 Teacher 统一走模型原生 chat template，只通过参数控制 thinking。
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    # rendered 已包含模板特殊 token，这里不再重复添加。
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("chat prompt token 为空")
    return prompt_ids


def build_student_prompt_ids(
    tokenizer: Any, problem: str, enable_thinking: bool
) -> list[int]:
    """构造只包含题目、不包含参考解答的 Student prompt。"""
    # Student 必须与实际推理条件一致，只能看到 problem。
    user_message = f"Problem: {problem.strip()}\n\n{STUDENT_INSTRUCTION}"
    return render_chat_prompt(tokenizer, user_message, enable_thinking)


def build_teacher_prompt_ids(
    tokenizer: Any,
    problem: str,
    solution: str,
    enable_thinking: bool,
) -> list[int]:
    """构造包含题目和特权参考解答的 Teacher prompt。"""
    # Teacher 额外看到 solution，并被要求理解后独立推导同一答案。
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


async def teacher_completion_logprobs_async(
    teacher_client: Any,
    teacher_prompt_ids: list[int],
    completion_ids: list[int],
    semaphore: asyncio.Semaphore,
) -> list[float]:
    """受并发限制地计算 Teacher 对 Student completion 的逐 token logprob。"""
    # Teacher 必须在特权 prompt 下评估 Student 已经生成的同一条轨迹。
    all_ids = teacher_prompt_ids + completion_ids
    # Student 采样和 Teacher 打分共享信号量，避免同时提交过多远程请求。
    async with semaphore:
        all_logprobs = await teacher_client.compute_logprobs_async(
            trio.ModelInput.from_ints(all_ids)
        )
    # prompt 区间不参与 OPSD，只保留 completion token 对应的 Teacher logprob。
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

    # 自回归训练使用 token[t] 预测 token[t+1]，因此输入末尾去掉最后一个 token。
    prompt_loss_len = len(student_prompt_ids) - 1
    input_ids = student_prompt_ids + completion_ids[:-1]
    # prompt 区间填零占位并屏蔽训练，只优化 Student 实际生成的 completion。
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


async def run_prompt_rollout_async(
    student_sampler: Any,
    teacher_client: Any,
    tokenizer: Any,
    row: dict[str, Any],
    args: argparse.Namespace,
    sampling_params: trio.SamplingParams,
    semaphore: asyncio.Semaphore,
) -> PromptRollout:
    """异步完成单题 Student 采样、Teacher 打分和 Datum 构造。"""
    # 每行训练数据必须同时提供公开 problem 和 Teacher 专用 solution。
    problem = str(row["problem"]).strip()
    solution = str(row["solution"]).strip()
    if not problem or not solution:
        raise ValueError("OPSD row 的 problem/solution 不能为空")

    # 两个角色使用同一 tokenizer，但看到的上下文不同。
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
    # Student rollout 属于远程 I/O，通过共享信号量限制整个 step 的并发量。
    async with semaphore:
        sample_result = await student_sampler.sample_async(
            prompt=trio.ModelInput.from_ints(student_prompt_ids),
            num_samples=args.group_size,
            sampling_params=sampling_params,
            return_text=False,
        )

    # 丢弃空 completion，并发请求 Teacher 对每条有效 Student 轨迹打分。
    sequences = [sequence for sequence in sample_result.sequences if sequence.tokens]
    teacher_logprobs_list = await asyncio.gather(
        *(
            teacher_completion_logprobs_async(
                teacher_client,
                teacher_prompt_ids,
                list(sequence.tokens),
                semaphore,
            )
            for sequence in sequences
        )
    )

    datums: list[trio.Datum] = []
    reverse_kls: list[float] = []
    student_logprobs: list[float] = []
    teacher_logprobs: list[float] = []
    completion_token_counts: list[int] = []
    sample_text = None

    # 每条 completion 独立计算 sampled-token reverse KL 和逐 token advantage。
    for sequence, teacher_lps in zip(
        sequences,
        teacher_logprobs_list,
        strict=True,
    ):
        completion_ids = list(sequence.tokens)
        if len(sequence.logprobs) != len(completion_ids) or any(
            value is None for value in sequence.logprobs
        ):
            raise ValueError("Student completion token/logprob 无法一一对齐")
        student_lps = [float(value) for value in sequence.logprobs]

        # reverse_kl 是 Student 采样分布下的单 token Monte Carlo 项。
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
        # 每道题仅保留第一条有效文本，用于低频 SwanLab 样例记录。
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
    # 只记录可转换为 float 的标量，跳过复杂对象或文本字段。
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
    # --no-swanlab 时完全跳过初始化，训练主循环无需感知具体运行模式。
    if not args.swanlab:
        return None
    # 将命令行参数和运行时补充信息一起保存，便于后续复现实验。
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
        tags=["PyTRIO", "OPSD", "OpenThoughts", "fixed-teacher", "async"],
        log_dir=str(SCRIPT_DIR / "swanlog"),
    )


async def save_checkpoint_async(
    training_client: Any,
    swanlab_run: Any | None,
    run_name: str,
    completed_steps: int,
) -> tuple[str, str]:
    """异步保存可续训 state 和可用于评测的 sampler weights。"""
    # 使用固定宽度 step 标签，确保 checkpoint 名称按训练顺序自然排序。
    tag = f"step{completed_steps:06d}"
    name = f"{run_name}-{tag}"

    # PyTRIO async 保存分为“提交任务”和“等待 APIFuture”两个 await。
    state_future = await training_client.save_state_async(name=name)
    state_result = await state_future
    weights_future = await training_client.save_weights_for_sampler_async(name=name)
    weights_result = await weights_future
    print(f"Saved state [{tag}]: {state_result.path}")
    print(f"Saved sampler weights [{tag}]: {weights_result.path}")
    if swanlab_run is not None:
        # checkpoint 日志也只在主协程写，和训练 step 顺序保持一致。
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
    # 未启用 SwanLab 时直接返回，避免 finally 阶段再次触发异常。
    if swanlab_run is None:
        return
    # 区分正常完成、用户中断和程序异常，方便在实验列表中筛选状态。
    if error is None:
        swanlab.finish()
    elif isinstance(error, KeyboardInterrupt):
        swanlab.finish(state="aborted")
    else:
        swanlab.finish(state="crashed", error=str(error))


async def train(args: argparse.Namespace) -> None:
    """执行异步 OPSD 主循环，并在主协程顺序记录日志与 checkpoint。"""
    # 固定本地随机源；数据 shuffle、远程训练和 Student 采样也复用同一 seed。
    random.seed(args.seed)
    np.random.seed(args.seed)
    dataset = load_training_dataset(args)

    # steps=0 表示完整遍历一次数据，否则严格执行命令行指定的 step 数。
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

    # ServiceClient 是创建异步训练客户端和采样客户端的统一入口。
    service_client = trio.ServiceClient()
    swanlab_run = None
    caught_error: BaseException | None = None
    try:
        # Student 只更新 LoRA 参数；attention 和 MLP 默认参与训练，unembed 可选。
        training_client = await service_client.create_lora_training_client_async(
            base_model=args.base_model,
            rank=args.lora_rank,
            seed=args.seed,
            train_attn=True,
            train_mlp=True,
            train_unembed=args.train_unembed,
        )
        # 不传 model_path，Teacher 始终保持为 step-0 base policy。
        teacher_client = await service_client.create_sampling_client_async(
            base_model=args.base_model
        )
        tokenizer = teacher_client.get_tokenizer()

        # 同一个 run 的所有 Student rollout 共用这一组采样参数。
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
        # 每个 step 聚合全部 Datum 后，只提交一次 optimizer update。
        adam = trio.AdamParams(
            learning_rate=args.learning_rate,
            beta1=args.beta1,
            beta2=args.beta2,
        )
        swanlab_run = start_swanlab(args, len(dataset), total_steps)

        # Student sampling 和 Teacher logprob 共用信号量，统一限制远程并发数。
        semaphore = asyncio.Semaphore(args.max_concurrency)
        student_sampler = None
        saved_steps: set[int] = set()

        for step in range(total_steps):
            step_start = time.time()

            # 刷新 sampler 权重；间隔为 1 时，每个 step 都使用最新 Student rollout。
            if student_sampler is None or step % args.sampler_refresh_steps == 0:
                student_sampler = (
                    await training_client.save_weights_and_get_sampling_client_async()
                )

            # 从训练开始时已经打乱的数据集中取得当前 problem batch。
            batch = batch_for_step(
                dataset,
                step,
                args.batch_size,
                full_dataset_run,
            )

            # 当前 step 的多个 problem 并发执行 rollout，进度条按题目完成数更新。
            with tqdm(
                total=len(batch),
                desc=f"OPSD async step {step + 1}",
                unit="problem",
            ) as progress:

                async def rollout_and_track(row: dict[str, Any]) -> PromptRollout:
                    """并发处理单题 rollout，并在完成后更新进度条。"""
                    # 单题内部仍受共享 semaphore 限制，不会突破最大远程并发数。
                    rollout = await run_prompt_rollout_async(
                        student_sampler,
                        teacher_client,
                        tokenizer,
                        row,
                        args,
                        sampling_params,
                        semaphore,
                    )
                    # 只在当前事件循环主线程更新 tqdm，避免并发输出互相覆盖。
                    progress.update(1)
                    return rollout

                # 等待当前 batch 的所有题目完成，之后才能构造统一训练 batch。
                rollouts = await asyncio.gather(
                    *(rollout_and_track(row) for row in batch)
                )

            # 展平题目级结果，准备一次远程 forward/backward 和指标聚合。
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

            # 第一次 await 提交请求并取得 APIFuture；第二次 await 等待远程任务结果。
            fwd_bwd_future = await training_client.forward_backward_async(
                datums,
                loss_fn="importance_sampling",
            )
            optim_future = await training_client.optim_step_async(adam)
            fwd_bwd_result = await fwd_bwd_future
            await optim_future

            # 远程更新完成后，再汇总本 step 的训练、OPSD 差异和整体耗时指标。
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
            # PyTRIO 返回的标量统一放进 trainer/* 命名空间。
            metrics.update(numeric_trainer_metrics(fwd_bwd_result))

            # 按配置抽取一条 Student completion，避免每个 step 上传大量文本。
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

            # 所有并发 rollout/teacher/training task 都已汇总；此处由主协程串行写 SwanLab。
            if swanlab_run is not None:
                swanlab.log(metrics, step=completed_steps)
            tqdm.write(
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
                await save_checkpoint_async(
                    training_client,
                    swanlab_run,
                    args.run_name,
                    completed_steps,
                )
                saved_steps.add(completed_steps)

        # 如果最后一步不在定期保存点上，额外保存最终双份 checkpoint。
        if total_steps not in saved_steps:
            await save_checkpoint_async(
                training_client,
                swanlab_run,
                args.run_name,
                total_steps,
            )
        print(f"Completed {total_steps} OPSD steps")
    except BaseException as error:
        # 保存异常对象交给 finally，由 SwanLab 标记 crashed 或 aborted 状态。
        caught_error = error
        raise
    finally:
        # 无论正常结束、用户中断还是异常，都保证 SwanLab run 正确关闭。
        finish_swanlab(swanlab_run, caught_error)


if __name__ == "__main__":
    started = time.time()
    asyncio.run(train(parse_args()))
    print(f"All done in {time.time() - started:.2f}s")
