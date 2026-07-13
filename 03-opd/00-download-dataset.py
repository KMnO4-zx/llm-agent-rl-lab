"""下载并整理中文医疗 SFT / 医疗评测 / 通用 OPD 数据。

默认会准备三类数据：
1. FreedomIntelligence/medical-o1-reasoning-SFT zh：医疗 reasoning SFT。
2. bigbio/med_qa med_qa_zh_4options_source：中文 MedQA 四选一评测。
3. ceval/ceval-exam：精选 8 个中文通用能力 subset，构造 OPD train pool 和 held-out test。

脚本会先完整下载/缓存原始数据，再按固定 seed 导出训练和评测文件。

默认规则：
- medical SFT 使用全量训练数据。
- MedQA-zh 从完整 test 中抽 600 条做医疗 eval。
- C-Eval 精选 subset 按 80% / 20% 切分，80% 做 OPD，剩余部分抽 300 条做 test。

正式下载：
uv run python 00-download-dataset.py

小成本调试：
uv run python 00-download-dataset.py \
    --medical-sft-sample-size 100 \
    --medqa-sample-size 20 \
    --ceval-train-size 100 \
    --ceval-test-size 20
"""

from __future__ import annotations

import argparse
import json
import math
import random
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import hf_hub_download


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "datasets"

MEDICAL_SFT_REPO = "FreedomIntelligence/medical-o1-reasoning-SFT"
MEDICAL_SFT_CONFIG = "zh"

MEDQA_REPO = "bigbio/med_qa"
MEDQA_CONFIG = "med_qa_zh_4options_source"

CEVAL_REPO = "ceval/ceval-exam"
CEVAL_EXCLUDED_CONFIGS = {
    "basic_medicine",
    "clinical_medicine",
    "physician",
    "veterinary_medicine",
    "high_school_biology",
    "middle_school_biology",
}

CEVAL_DEFAULT_CONFIGS = [
    "computer_network",
    "college_programming",
    "advanced_mathematics",
    "discrete_mathematics",
    "college_physics",
    "logic",
    "chinese_language_and_literature",
    "college_economics",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载中文医疗 SFT、MedQA-zh eval 和 C-Eval OPD/eval 数据")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="数据输出目录")
    parser.add_argument("--cache-dir", type=Path, default=None, help="HuggingFace datasets cache 目录")
    parser.add_argument("--seed", type=int, default=42, help="固定抽样和切分使用的随机种子")

    parser.add_argument("--skip-medical-sft", action="store_true", help="跳过 medical-o1 reasoning SFT 数据")
    parser.add_argument("--skip-medqa", action="store_true", help="跳过 MedQA-zh 评测数据")
    parser.add_argument("--skip-ceval", action="store_true", help="跳过 C-Eval 通用数据")

    parser.add_argument("--medical-sft-sample-size", type=int, default=0, help="调试用医疗 SFT sample 数量；<=0 表示不单独导出 sample，SFT 默认使用全量 train.jsonl")
    parser.add_argument("--medqa-sample-size", type=int, default=600, help="从完整 MedQA-zh test 中导出的 eval sample 数量；<=0 表示全量")
    parser.add_argument("--ceval-test-size", "--ceval-eval-size", dest="ceval_test_size", type=int, default=300, help="从 C-Eval held-out test pool 中导出的 test sample 数量")
    parser.add_argument("--ceval-train-ratio", type=float, default=0.8, help="C-Eval 数据中用于 OPD train pool 的比例")
    parser.add_argument("--ceval-train-size", type=int, default=0, help="从 C-Eval OPD train pool 中导出的 OPD train 数量；<=0 表示使用完整 train pool")
    parser.add_argument(
        "--ceval-splits",
        nargs="+",
        default=["dev", "val", "test"],
        help="参与 C-Eval 80/20 切分的 split",
    )
    parser.add_argument("--ceval-configs", nargs="*", default=None, help="指定 C-Eval config；不传表示使用默认精选 8 个 subset")
    args = parser.parse_args()
    if not 0 < args.ceval_train_ratio < 1:
        raise ValueError("--ceval-train-ratio must be between 0 and 1")
    return args


def cache_dir(args: argparse.Namespace) -> str:
    path = args.cache_dir or (args.data_dir / ".hf_cache")
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def shuffled_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows


def select_size(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    if size <= 0:
        return rows
    return rows[: min(size, len(rows))]


def first_text(row: dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def medical_sft_completion(complex_cot: str, response: str) -> str:
    # 数据文件保留原始 reasoning + answer；模型专用的 <think> 标签在 SFT 脚本里构造。
    parts = [part.strip() for part in (complex_cot, response) if part and part.strip()]
    return "\n\n".join(parts)


def normalize_medical_sft_row(row: dict[str, Any]) -> dict[str, Any] | None:
    question = first_text(row, ("Question", "question", "instruction", "prompt"))
    complex_cot = first_text(row, ("Complex_CoT", "complex_cot", "cot", "reasoning"))
    response = first_text(row, ("Response", "response", "answer", "output"))
    completion = medical_sft_completion(complex_cot, response)
    if not question or not completion:
        return None
    return {
        "question": question,
        "complex_cot": complex_cot,
        "response": response,
        "completion": completion,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": completion},
        ],
    }


def prepare_medical_sft(args: argparse.Namespace, hf_cache_dir: str) -> None:
    print(f"Loading {MEDICAL_SFT_REPO}/{MEDICAL_SFT_CONFIG} ...")
    data = load_dataset(MEDICAL_SFT_REPO, MEDICAL_SFT_CONFIG, cache_dir=hf_cache_dir)
    if not isinstance(data, DatasetDict):
        data = DatasetDict({"train": data})

    out_dir = args.data_dir / "medical-o1-reasoning-SFT-zh"
    for split, dataset in data.items():
        rows = []
        for row in dataset:
            normalized = normalize_medical_sft_row(dict(row))
            if normalized is not None:
                normalized["source_split"] = split
                rows.append(normalized)
        rows = shuffled_rows(rows, args.seed)
        all_path = out_dir / f"{split}_all.jsonl"
        all_count = write_jsonl(all_path, rows)
        print(f"Saved medical SFT {split} all: {all_count} rows -> {all_path}")

        if split == "train":
            train_path = out_dir / "train.jsonl"
            train_count = write_jsonl(train_path, rows)
            print(f"Saved medical SFT train: {train_count} rows -> {train_path}")
            if args.medical_sft_sample_size > 0:
                sample_rows = select_size(rows, args.medical_sft_sample_size)
                sample_path = out_dir / "train_sample.jsonl"
                sample_count = write_jsonl(sample_path, sample_rows)
                print(f"Saved medical SFT train sample: {sample_count} rows -> {sample_path}")


def option_dict_from_any(options: Any) -> dict[str, str]:
    if isinstance(options, dict):
        return {str(k).upper(): str(v).strip() for k, v in options.items() if str(v).strip()}
    if isinstance(options, list):
        result: dict[str, str] = {}
        for i, item in enumerate(options):
            default_key = chr(ord("A") + i)
            if isinstance(item, dict):
                key = item.get("key") or item.get("label") or item.get("id") or item.get("name") or default_key
                value = item.get("value") or item.get("text") or item.get("option") or item.get("content")
            else:
                key = default_key
                value = item
            if value is not None and str(value).strip():
                result[str(key).upper()] = str(value).strip()
        return result
    return {}


def option_dict_from_abcd(row: dict[str, Any]) -> dict[str, str]:
    options = {}
    for key in ("A", "B", "C", "D"):
        value = row.get(key)
        if value is not None and str(value).strip():
            options[key] = str(value).strip()
    return options


def normalize_answer_idx(answer_idx: Any, options: dict[str, str], answer: str) -> str:
    if answer_idx is not None and str(answer_idx).strip():
        value = str(answer_idx).strip().upper()
        if value in options:
            return value
    for key, value in options.items():
        if answer and value.strip() == answer.strip():
            return key
    return ""


def choice_prompt(question: str, options: dict[str, str]) -> str:
    option_lines = [f"{key}. {options[key]}" for key in ("A", "B", "C", "D") if key in options]
    return "\n".join(
        [
            "以下是中国考试中的单项选择题。请仔细思考，并只输出最终答案选项字母。",
            "",
            f"题目：{question}",
            *option_lines,
            "",
            "答案：",
        ]
    )


def normalize_medqa_row(row: dict[str, Any]) -> dict[str, Any] | None:
    question = first_text(row, ("question", "Question"))
    options = option_dict_from_any(row.get("options")) or option_dict_from_abcd(row)
    answer = first_text(row, ("answer", "Answer"))
    answer_idx = normalize_answer_idx(row.get("answer_idx") or row.get("answer_index"), options, answer)
    if not question or len(options) < 4 or not answer_idx:
        return None
    return {
        "question": question,
        "options": {key: options[key] for key in ("A", "B", "C", "D") if key in options},
        "answer": answer,
        "answer_idx": answer_idx,
        "prompt": choice_prompt(question, options),
    }


def prepare_medqa(args: argparse.Namespace, hf_cache_dir: str) -> None:
    print(f"Loading {MEDQA_REPO}/{MEDQA_CONFIG} test ...")
    # bigbio/med_qa 仍保留旧版 datasets loading script；新版 datasets 不再执行这类脚本。
    # 这里直接下载仓库里的 data_clean.zip，再读取中文 Mainland/4_options/test.jsonl。
    zip_path = hf_hub_download(
        repo_id=MEDQA_REPO,
        filename="data_clean.zip",
        repo_type="dataset",
        cache_dir=hf_cache_dir,
    )
    target_suffix = "data_clean/questions/Mainland/4_options/test.jsonl"
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        matched_names = [name for name in zf.namelist() if name.endswith(target_suffix)]
        if not matched_names:
            raise FileNotFoundError(f"Cannot find {target_suffix} in {zip_path}")
        with zf.open(matched_names[0]) as f:
            for line in f:
                row = json.loads(line.decode("utf-8"))
                normalized = normalize_medqa_row(row)
                if normalized is not None:
                    rows.append(normalized)
    rows = shuffled_rows(rows, args.seed)

    out_dir = args.data_dir / "medqa-zh-4options"
    all_path = out_dir / "test_all.jsonl"
    all_count = write_jsonl(all_path, rows)
    print(f"Saved MedQA-zh test all: {all_count} rows -> {all_path}")

    sample_rows = select_size(rows, args.medqa_sample_size)
    sample_path = out_dir / "test_sample.jsonl"
    sample_count = write_jsonl(sample_path, sample_rows)
    print(f"Saved MedQA-zh eval sample: {sample_count} rows -> {sample_path}")


def ceval_configs(args: argparse.Namespace, hf_cache_dir: str) -> list[str]:
    if args.ceval_configs:
        configs = list(args.ceval_configs)
    else:
        configs = CEVAL_DEFAULT_CONFIGS
    return [name for name in configs if name not in CEVAL_EXCLUDED_CONFIGS]


def normalize_ceval_row(row: dict[str, Any], subject: str, split: str, row_index: int) -> dict[str, Any] | None:
    question = first_text(row, ("question", "Question"))
    options = option_dict_from_abcd(row)
    answer = first_text(row, ("answer", "Answer"))
    answer_idx = normalize_answer_idx(answer, options, answer="")
    if not question or len(options) < 4:
        return None
    return {
        "row_id": f"{subject}:{split}:{row_index}",
        "subject": subject,
        "source_split": split,
        "question": question,
        "options": options,
        "answer_idx": answer_idx,
        "prompt": choice_prompt(question, options),
    }


def load_ceval_split(config: str, split: str, hf_cache_dir: str) -> Dataset | None:
    try:
        return load_dataset(CEVAL_REPO, config, split=split, cache_dir=hf_cache_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"Skip C-Eval {config}/{split}: {exc}")
        return None


def prepare_ceval(args: argparse.Namespace, hf_cache_dir: str) -> None:
    configs = ceval_configs(args, hf_cache_dir)
    print(f"Loading C-Eval non-med configs: {len(configs)} subjects")

    all_rows: list[dict[str, Any]] = []
    loaded_configs: set[str] = set()
    skipped_splits: list[str] = []
    for config in configs:
        config_loaded = False
        for split in args.ceval_splits:
            dataset = load_ceval_split(config, split, hf_cache_dir)
            if dataset is None:
                skipped_splits.append(f"{config}/{split}")
                continue
            config_loaded = True
            for row_index, row in enumerate(dataset):
                normalized = normalize_ceval_row(dict(row), config, split, row_index)
                if normalized is not None:
                    all_rows.append(normalized)
        if config_loaded:
            loaded_configs.add(config)

    missing_configs = sorted(set(configs) - loaded_configs)
    if missing_configs:
        raise RuntimeError(
            "Some C-Eval configs were not loaded. "
            f"Missing: {missing_configs}. "
            "Check network/cache, or pass --ceval-configs with only available configs."
        )

    all_rows = shuffled_rows(all_rows, args.seed)
    answer_rows = [row for row in all_rows if row["answer_idx"]]
    no_answer_rows = [row for row in all_rows if not row["answer_idx"]]

    heldout_size = math.ceil(len(answer_rows) * (1 - args.ceval_train_ratio))
    heldout_size = min(heldout_size, len(answer_rows))

    test_pool = answer_rows[:heldout_size]
    train_pool = shuffled_rows(answer_rows[heldout_size:] + no_answer_rows, args.seed + 1)
    train_rows = select_size(train_pool, args.ceval_train_size)
    test_rows = select_size(test_pool, args.ceval_test_size)

    out_dir = args.data_dir / "ceval-non-med"
    all_count = write_jsonl(out_dir / "all.jsonl", all_rows)
    test_pool_count = write_jsonl(out_dir / "test_pool.jsonl", test_pool)
    train_pool_count = write_jsonl(out_dir / "train_pool.jsonl", train_pool)
    test_count = write_jsonl(out_dir / "test_sample.jsonl", test_rows)
    train_count = write_jsonl(out_dir / "opd_train.jsonl", train_rows)
    write_json(
        out_dir / "manifest.json",
        {
            "repo": CEVAL_REPO,
            "excluded_configs": sorted(CEVAL_EXCLUDED_CONFIGS),
            "configs": configs,
            "seed": args.seed,
            "splits": args.ceval_splits,
            "train_ratio": args.ceval_train_ratio,
            "loaded_configs": sorted(loaded_configs),
            "skipped_splits": sorted(skipped_splits),
            "all_rows": all_count,
            "test_pool_rows": test_pool_count,
            "train_pool_rows": train_pool_count,
            "test_sample_rows": test_count,
            "opd_train_rows": train_count,
        },
    )
    print(f"Saved C-Eval all rows: {all_count} rows -> {out_dir / 'all.jsonl'}")
    print(f"Saved C-Eval OPD train pool: {train_pool_count} rows -> {out_dir / 'train_pool.jsonl'}")
    print(f"Saved C-Eval held-out test pool: {test_pool_count} rows -> {out_dir / 'test_pool.jsonl'}")
    print(f"Saved C-Eval OPD train: {train_count} rows -> {out_dir / 'opd_train.jsonl'}")
    print(f"Saved C-Eval general test: {test_count} rows -> {out_dir / 'test_sample.jsonl'}")


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    hf_cache_dir = cache_dir(args)

    if not args.skip_medical_sft:
        prepare_medical_sft(args, hf_cache_dir)
    if not args.skip_medqa:
        prepare_medqa(args, hf_cache_dir)
    if not args.skip_ceval:
        prepare_ceval(args, hf_cache_dir)

    print("All requested datasets are ready.")


if __name__ == "__main__":
    main()
