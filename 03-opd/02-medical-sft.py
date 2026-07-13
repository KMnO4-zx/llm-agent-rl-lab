"""中文医疗 reasoning SFT：medical-o1-reasoning-SFT-zh + PyTRIO async + SwanLab。

训练格式：
- prompt 只包含 system + user，并用 chat template 加 generation prompt。
- assistant completion 由脚本构造成 <think>{Complex_CoT}</think> + Response。
- prompt token 不参与 loss；assistant completion token 和 EOS 参与 loss。
- checkpoint 同时保存 training state 和 sampler weights；state 用于后续 OPD 初始化 student，sampler weights 用于评测。

小成本试跑：
uv run python 02-medical-sft.py \
    --sample-size 100 \
    --num-epochs 1 \
    --batch-size 2 \
    --max-length 2048 \
    --swanlab-mode disabled

正式训练：
uv run python 02-medical-sft.py \
    --num-epochs 3 \
    --batch-size 16 \
    --max-length 2048 \
    --swanlab-mode online
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
DEFAULT_DATASET_PATH = SCRIPT_DIR / "datasets" / "medical-o1-reasoning-SFT-zh" / "train.jsonl"
DEFAULT_SYSTEM_MESSAGE = "你是一个中文医疗问答助手。请根据题目给出严谨的医学推理和最终答案。"


@dataclass(frozen=True)
class TokenizedExample:
    tokens: list[int]
    weights: list[float]
    prompt_len: int
    truncated: bool


@dataclass(frozen=True)
class ProcessedDatum:
    datum: trio.Datum
    total_tokens: int
    prompt_tokens: int
    trainable_tokens: int
    truncated: bool


def model_slug(base_model: str) -> str:
    name = base_model.rsplit("/", 1)[-1].lower()
    name = name.replace("qwen3.5", "qwen35")
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def build_run_name(args: argparse.Namespace) -> str:
    dataset_slug = args.dataset_path.parent.name.lower().replace("_", "-")
    dataset_slug = re.sub(r"[^a-z0-9-]+", "-", dataset_slug).strip("-")
    sample_slug = f"sample{args.sample_size}" if args.sample_size > 0 else "full"
    steps_slug = f"-steps{args.max_steps}" if args.max_steps > 0 else ""
    return f"sft-medical-{model_slug(args.base_model)}-{dataset_slug}-{sample_slug}-epochs{args.num_epochs}{steps_slug}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTRIO 异步版中文医疗 SFT")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH, help="medical-o1 reasoning SFT JSONL 路径")
    parser.add_argument("--sample-size", type=int, default=0, help="随机抽样数量；<=0 表示使用全量训练集")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B", help="LoRA SFT 使用的基础模型")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--num-epochs", type=int, default=1, help="训练 epoch 数")
    parser.add_argument("--batch-size", type=int, default=16, help="每个 step 的样本数")
    parser.add_argument("--max-steps", type=int, default=0, help="最多训练 step 数；<=0 表示不限制")
    parser.add_argument("--max-length", type=int, default=2048, help="单条样本 prompt+completion 最大 token 数")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam 学习率")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2")
    parser.add_argument("--system-message", default=DEFAULT_SYSTEM_MESSAGE, help="训练时插入到 chat template 的 system message；传空字符串表示不加")
    parser.add_argument("--save-each-epoch", action=argparse.BooleanOptionalAction, default=True, help="是否每个 epoch 结束同时保存 training state 和 sampler 权重")
    parser.add_argument("--save-every-steps", type=int, default=1000, help="每多少 step 同时保存 training state 和 sampler 权重；0 表示不按 step 保存")

    parser.add_argument("--swanlab", action=argparse.BooleanOptionalAction, default=True, help="是否启用 SwanLab 记录")
    parser.add_argument("--swanlab-project", default="llm-agent-rl-lab", help="SwanLab 项目名")
    parser.add_argument("--swanlab-workspace", default=None, help="SwanLab workspace；不传则使用默认 workspace")
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "offline", "disabled"],
        default=None,
        help="SwanLab 运行模式",
    )
    args = parser.parse_args()

    for name in ("num_epochs", "batch_size", "max_length"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.sample_size < 0:
        raise ValueError("--sample-size must be >= 0")
    if args.max_steps < 0:
        raise ValueError("--max-steps must be >= 0")
    if args.save_every_steps < 0:
        raise ValueError("--save-every-steps must be >= 0")

    run_name = build_run_name(args)
    args.swanlab_name = run_name
    args.save_state_name = run_name
    args.save_weights_name = run_name
    return args


def build_assistant_text(complex_cot: str, response: str) -> str:
    complex_cot = complex_cot.strip()
    response = response.strip()
    # medical-o1 把推理过程和最终回答分成两个字段；
    # 训练 Qwen thinking 模型时，把推理过程包进 <think>...</think>。
    if complex_cot and response:
        return f"<think>\n{complex_cot}\n</think>\n\n{response}"
    if complex_cot:
        return f"<think>\n{complex_cot}\n</think>"
    return response


def normalize_row(row: dict[str, Any]) -> dict[str, str] | None:
    question = str(row.get("question", "")).strip()
    complex_cot = str(row.get("complex_cot", "")).strip()
    response = str(row.get("response", "")).strip()
    assistant = build_assistant_text(complex_cot, response)
    if not question or not assistant:
        return None
    return {
        "question": question,
        "assistant": assistant,
    }


def load_examples(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.dataset_path.exists():
        raise FileNotFoundError(f"找不到 SFT 数据文件：{args.dataset_path}，请先运行 00-download-dataset.py")

    examples = []
    with args.dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            example = normalize_row(json.loads(line))
            if example is not None:
                examples.append(example)

    if not examples:
        raise ValueError(f"没有在 {args.dataset_path} 中找到有效 SFT 样本")

    random.Random(args.seed).shuffle(examples)
    if args.sample_size > 0:
        examples = examples[: min(args.sample_size, len(examples))]
    return examples


def build_prompt_tokens(tokenizer: Any, question: str, system_message: str) -> list[int]:
    messages = []
    if system_message.strip():
        messages.append({"role": "system", "content": system_message.strip()})
    messages.append({"role": "user", "content": question})

    # 这里只把 system + user 放进 chat template，并让模板补出 assistant 起始位置。
    # assistant 正文会在后面单独 tokenize 后拼接，不作为 messages 输入模板。
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer.encode(prompt_text, add_special_tokens=False)


def tokenize_example(
    example: dict[str, str],
    tokenizer: Any,
    args: argparse.Namespace,
) -> TokenizedExample:
    prompt_tokens = build_prompt_tokens(tokenizer, example["question"], args.system_message)
    assistant_tokens = tokenizer.encode(example["assistant"], add_special_tokens=False)

    # 让模型学习回答结束位置；EOS 也属于 assistant completion 的训练目标。
    if tokenizer.eos_token_id is not None:
        assistant_tokens.append(int(tokenizer.eos_token_id))

    tokens = prompt_tokens + assistant_tokens

    # prompt 只是上下文，不参与 loss；assistant completion 和 EOS 参与 loss。
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(assistant_tokens)
    truncated = len(tokens) > args.max_length
    if truncated:
        tokens = tokens[: args.max_length]
        weights = weights[: args.max_length]

    prompt_len = min(len(prompt_tokens), len(tokens))
    return TokenizedExample(
        tokens=tokens,
        weights=weights,
        prompt_len=prompt_len,
        truncated=truncated,
    )


def build_datum(
    example: dict[str, str],
    tokenizer: Any,
    args: argparse.Namespace,
) -> ProcessedDatum | None:
    tokenized = tokenize_example(example, tokenizer, args)
    tokens = tokenized.tokens
    weights = tokenized.weights

    if len(tokens) < 2:
        return None

    # 自回归右移：当前位置 input 预测下一个 target。
    # weights 也右移到 target 侧，因此第一个 assistant token 会被正确训练到。
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    loss_weights = weights[1:]
    if sum(loss_weights) <= 0:
        return None

    datum = trio.Datum(
        model_input=trio.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_tokens, dtype=np.int32),
            "weights": np.asarray(loss_weights, dtype=np.float32),
        },
    )
    return ProcessedDatum(
        datum=datum,
        total_tokens=len(tokens),
        prompt_tokens=tokenized.prompt_len,
        trainable_tokens=int(sum(loss_weights)),
        truncated=tokenized.truncated,
    )


def build_processed_examples(
    examples: list[dict[str, str]],
    tokenizer: Any,
    args: argparse.Namespace,
) -> list[ProcessedDatum]:
    processed_examples = []
    for example in tqdm(examples, desc="Build SFT datums", unit="sample"):
        built = build_datum(example, tokenizer, args)
        if built is not None:
            processed_examples.append(built)

    if not processed_examples:
        raise RuntimeError("No tokenized SFT example has trainable assistant tokens")
    return processed_examples


def iter_epoch_batches(
    examples: list[ProcessedDatum],
    batch_size: int,
    seed: int,
    epoch: int,
) -> list[list[ProcessedDatum]]:
    indices = list(range(len(examples)))
    random.Random(seed + epoch).shuffle(indices)
    return [
        [examples[index] for index in indices[start : start + batch_size]]
        for start in range(0, len(indices), batch_size)
    ]


def to_float_array(value: Any) -> np.ndarray:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return np.asarray(value, dtype=np.float32)


def weighted_loss(fwdbwd_result: Any, datums: list[trio.Datum]) -> float:
    # TRIO 返回的是每个 target token 的 logprob；这里按 assistant-only weights 还原平均 CE loss。
    logprobs = np.concatenate(
        [to_float_array(output["logprobs"]) for output in fwdbwd_result.loss_fn_outputs]
    )
    weights = np.concatenate(
        [to_float_array(datum.loss_fn_inputs["weights"]) for datum in datums]
    )
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return 0.0
    return float(-np.dot(logprobs, weights) / weight_sum)


def length_stats(values: list[int], prefix: str) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float32)
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_max": int(arr.max()),
    }


def analyze_dataset_tokens(
    processed_examples: list[ProcessedDatum],
    sample_count: int,
) -> dict[str, float | int]:
    # 这里直接复用训练前已经构造好的 Datum 统计，保证统计路径和训练路径完全一致。
    context_lengths = [example.total_tokens for example in processed_examples]
    prompt_lengths = [example.prompt_tokens for example in processed_examples]
    trainable_lengths = [example.trainable_tokens for example in processed_examples]
    truncated_count = sum(1 for example in processed_examples if example.truncated)
    skipped_count = sample_count - len(processed_examples)
    valid_count = len(context_lengths)
    stats: dict[str, float | int] = {
        "sample_count": sample_count,
        "valid_sample_count": valid_count,
        "skipped_sample_count": skipped_count,
        "truncated_sample_count": truncated_count,
        "truncated_sample_rate": truncated_count / valid_count,
    }
    stats.update(length_stats(context_lengths, "context_tokens"))
    stats.update(length_stats(prompt_lengths, "prompt_tokens"))
    stats.update(length_stats(trainable_lengths, "trainable_tokens"))
    return stats


def print_dataset_token_stats(stats: dict[str, float | int]) -> None:
    print("=" * 25 + "Dataset token stats" + "=" * 25)
    print(
        f"- 样本数: {stats['valid_sample_count']}/{stats['sample_count']} | "
        f"跳过 {stats['skipped_sample_count']} | "
        f"截断 {stats['truncated_sample_count']} ({stats['truncated_sample_rate']:.2%})"
    )
    print(
        f"- 总 context tokens(prompt+assistant+EOS): mean {stats['context_tokens_mean']:.1f} | "
        f"p50 {stats['context_tokens_p50']:.1f} | p95 {stats['context_tokens_p95']:.1f} | "
        f"max {stats['context_tokens_max']}"
    )
    print(
        f"- prompt tokens(system+user): mean {stats['prompt_tokens_mean']:.1f} | "
        f"p50 {stats['prompt_tokens_p50']:.1f} | p95 {stats['prompt_tokens_p95']:.1f} | "
        f"max {stats['prompt_tokens_max']}"
    )
    print(
        f"- 可训练 tokens(weights=1): mean {stats['trainable_tokens_mean']:.1f} | "
        f"p50 {stats['trainable_tokens_p50']:.1f} | p95 {stats['trainable_tokens_p95']:.1f} | "
        f"max {stats['trainable_tokens_max']}"
    )
    print("=" * 22+ "End of dataset token stats" + "=" * 21)


async def log_step_result(
    fwdbwd_future: Any,
    optim_future: Any,
    batch: list[ProcessedDatum],
    args: argparse.Namespace,
    swanlab_run: Any | None,
    progress_bar: tqdm,
    step_start: float,
    step: int,
    total_steps: int,
    epoch: int,
) -> None:
    fwdbwd_result = await fwdbwd_future
    await optim_future

    datums = [item.datum for item in batch]
    total_tokens = int(sum(item.total_tokens for item in batch))
    truncated = int(sum(1 for item in batch if item.truncated))
    step_elapsed = time.time() - step_start
    loss = weighted_loss(fwdbwd_result, datums)

    metrics = {
        "train/loss": loss,
        "train/learning_rate": args.learning_rate,
        "train/epoch": epoch + 1,
        "train/step": step,
        "data/batch_size": len(datums),
        "data/total_tokens": total_tokens,
        "data/truncated_examples": truncated,
        "time/step_elapsed_time": step_elapsed,
        "time/tokens_per_second": total_tokens / step_elapsed,
    }
    metrics.update({f"trainer/{key}": float(value) for key, value in dict(fwdbwd_result.metrics).items()})
    if swanlab_run is not None:
        swanlab.log(metrics, step=step)

    progress_bar.update(1)
    progress_bar.set_postfix(loss=f"{loss:.4f}", epoch=f"{epoch + 1}/{args.num_epochs}")
    tqdm.write(
        f"step {step:04d}/{total_steps} | epoch {epoch + 1}/{args.num_epochs} | "
        f"loss {loss:.4f} | batch {len(datums)} | time {step_elapsed:.2f}s"
    )


def start_swanlab(
    args: argparse.Namespace,
    dataset_size: int,
    total_steps: int,
    token_stats: dict[str, float | int],
) -> Any | None:
    if not args.swanlab:
        return None
    config = vars(args).copy()
    config["dataset_path"] = str(args.dataset_path)
    config["dataset_size"] = dataset_size
    config["total_steps"] = total_steps
    config["run_name"] = args.swanlab_name
    config["state_name"] = args.save_state_name
    config["weights_name"] = args.save_weights_name
    config["dataset_token_stats"] = token_stats
    return swanlab.init(
        project=args.swanlab_project,
        name=args.swanlab_name,
        workspace=args.swanlab_workspace,
        mode=args.swanlab_mode,
        config=config,
        tags=["PyTrio", "SFT", "medical", "async"],
        log_dir=str(SCRIPT_DIR / "swanlog"),
    )


async def save_training_state(
    training_client: Any,
    swanlab_run: Any | None,
    name: str,
    step: int,
    tag: str,
) -> str:
    # save_state 保存完整训练状态；后续 OPD 可用 create_training_client_from_state_async 初始化 student。
    save_future = await training_client.save_state_async(name=name)
    save_result = await save_future
    print(f"Saved state [{tag}]: {save_result.path}")
    if swanlab_run is not None:
        swanlab.log({f"save/{tag}_state_path": swanlab.Text(save_result.path)}, step=step)
    return save_result.path


async def save_sampler_weights(
    training_client: Any,
    swanlab_run: Any | None,
    name: str,
    step: int,
    tag: str,
) -> str:
    save_future = await training_client.save_weights_for_sampler_async(name=name)
    save_result = await save_future
    print(f"Saved weights [{tag}]: {save_result.path}")
    if swanlab_run is not None:
        swanlab.log({f"save/{tag}_weights_path": swanlab.Text(save_result.path)}, step=step)
    return save_result.path


async def save_checkpoint(
    training_client: Any,
    swanlab_run: Any | None,
    state_name: str,
    weights_name: str,
    step: int,
    tag: str,
) -> tuple[str, str]:
    state_path = await save_training_state(
        training_client=training_client,
        swanlab_run=swanlab_run,
        name=state_name,
        step=step,
        tag=tag,
    )
    weights_path = await save_sampler_weights(
        training_client=training_client,
        swanlab_run=swanlab_run,
        name=weights_name,
        step=step,
        tag=tag,
    )
    return state_path, weights_path


async def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)

    examples = load_examples(args)
    print(f"Loaded {len(examples)} medical SFT examples")
    print(f"Run name: {args.swanlab_name}")
    print(f"State name: {args.save_state_name}")
    print(f"Sampler weights name: {args.save_weights_name}")

    service_client = trio.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=args.base_model,
        rank=args.lora_rank,
        seed=args.seed,
    )
    tokenizer = training_client.get_tokenizer()
    processed_examples = build_processed_examples(examples, tokenizer, args)
    print(f"Prepared {len(processed_examples)} valid SFT datums")

    steps_per_epoch = (len(processed_examples) + args.batch_size - 1) // args.batch_size
    planned_steps = args.num_epochs * steps_per_epoch
    total_steps = min(planned_steps, args.max_steps) if args.max_steps > 0 else planned_steps

    token_stats = analyze_dataset_tokens(processed_examples, len(examples))
    print_dataset_token_stats(token_stats)

    adam_params = trio.AdamParams(
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
    )

    swanlab_run = start_swanlab(args, len(processed_examples), total_steps, token_stats)
    submitted_steps = 0
    last_checkpoint_step = 0

    try:
        with tqdm(total=total_steps, desc="Medical SFT async", unit="step") as progress_bar:
            for epoch in range(args.num_epochs):
                remaining_steps = total_steps - submitted_steps
                if remaining_steps <= 0:
                    break

                epoch_submit_steps = min(steps_per_epoch, remaining_steps)
                log_tasks = []
                submit_bar = tqdm(
                    total=epoch_submit_steps,
                    desc=f"Epoch {epoch + 1}/{args.num_epochs} submit",
                    unit="step",
                    leave=False,
                )

                for batch in iter_epoch_batches(processed_examples, args.batch_size, args.seed, epoch):
                    if submitted_steps >= total_steps:
                        break

                    step_start = time.time()
                    step = submitted_steps
                    datums = [item.datum for item in batch]

                    # 连续提交远程训练任务，loss/SwanLab 记录放到后台 task。
                    fwdbwd_future = await training_client.forward_backward_async(
                        datums,
                        loss_fn="cross_entropy",
                    )
                    optim_future = await training_client.optim_step_async(adam_params)
                    submit_bar.update(1)
                    log_tasks.append(
                        asyncio.create_task(
                            log_step_result(
                                fwdbwd_future=fwdbwd_future,
                                optim_future=optim_future,
                                batch=batch,
                                args=args,
                                swanlab_run=swanlab_run,
                                progress_bar=progress_bar,
                                step_start=step_start,
                                step=step,
                                total_steps=total_steps,
                                epoch=epoch,
                            )
                        )
                    )
                    submitted_steps += 1
                    if args.save_every_steps > 0 and submitted_steps % args.save_every_steps == 0:
                        # 按 step 保存前，先等已提交任务完成，确保 checkpoint 对齐到明确的 step。
                        await asyncio.gather(*log_tasks)
                        log_tasks = []
                        await save_checkpoint(
                            training_client=training_client,
                            swanlab_run=swanlab_run,
                            state_name=f"{args.save_state_name}-step{submitted_steps:06d}",
                            weights_name=f"{args.save_weights_name}-step{submitted_steps:06d}",
                            step=submitted_steps,
                            tag=f"step{submitted_steps:06d}",
                        )
                        last_checkpoint_step = submitted_steps

                submit_bar.close()
                if log_tasks:
                    await asyncio.gather(*log_tasks)
                if args.save_each_epoch and submitted_steps % steps_per_epoch == 0:
                    await save_checkpoint(
                        training_client=training_client,
                        swanlab_run=swanlab_run,
                        state_name=f"{args.save_state_name}-epoch{epoch + 1:03d}",
                        weights_name=f"{args.save_weights_name}-epoch{epoch + 1:03d}",
                        step=submitted_steps,
                        tag=f"epoch{epoch + 1:03d}",
                    )
                    last_checkpoint_step = submitted_steps

                if submitted_steps >= total_steps:
                    break

        if submitted_steps == 0:
            raise RuntimeError("No SFT training step was completed")
        if last_checkpoint_step != submitted_steps:
            await save_checkpoint(
                training_client=training_client,
                swanlab_run=swanlab_run,
                state_name=f"{args.save_state_name}-step{submitted_steps:06d}",
                weights_name=f"{args.save_weights_name}-step{submitted_steps:06d}",
                step=submitted_steps,
                tag=f"step{submitted_steps:06d}",
            )
        print(f"Completed {submitted_steps} SFT steps")
    finally:
        if swanlab_run is not None:
            swanlab.finish()


def main() -> None:
    start = time.time()
    asyncio.run(train(parse_args()))
    print("#" * 50)
    print("# all done")
    print(f"# train cost {time.time() - start:.2f}s")
    print("#" * 50)


if __name__ == "__main__":
    main()
