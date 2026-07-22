"""从 Hugging Face 下载 OPSD 训练集与 AIME 2025 评测集。

默认下载两份数据并保存为 Hugging Face ``save_to_disk`` 格式：

    uv run python 00-datasets.py

只下载其中一份：

    uv run python 00-datasets.py --only opsd
    uv run python 00-datasets.py --only aime25

数据 revision 被固定，避免上游更新导致训练集或 benchmark 静默变化。
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR / "datasets"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    repo_id: str
    revision: str
    split: str
    output_name: str
    expected_rows: int
    required_columns: tuple[str, ...]


DATASETS = {
    "opsd": DatasetSpec(
        key="opsd",
        repo_id="siyanzhao/Openthoughts_math_30k_opsd",
        revision="1f33e9dc2e8a1c639ca74f8024ad4a9f1f5eae62",
        split="train",
        output_name="openthoughts_math_30k_opsd",
        expected_rows=29_434,
        required_columns=("problem", "solution"),
    ),
    "aime25": DatasetSpec(
        key="aime25",
        repo_id="yentinglin/aime_2025",
        revision="6f71d77b0b89b9dabe07ab466c51df33f514df7f",
        split="train",
        output_name="aime_2025",
        expected_rows=30,
        required_columns=("problem", "answer"),
    ),
}


def parse_args() -> argparse.Namespace:
    """解析数据集选择、缓存目录和强制重下等命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=["all", *DATASETS],
        default="all",
        help="下载全部数据，或只下载 OPSD/AIME25；默认 all",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="save_to_disk 数据保存目录",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Hugging Face 下载缓存；默认使用 <data-root>/.cache",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="删除并重新下载已经存在的目标数据目录",
    )
    return parser.parse_args()


def as_single_split(dataset: Dataset | DatasetDict, split: str) -> Dataset:
    """把单个 Dataset 或 DatasetDict 统一转换为指定 split。"""
    if isinstance(dataset, Dataset):
        return dataset
    if split not in dataset:
        raise ValueError(f"数据集中不存在 {split!r} split，可用 split: {list(dataset)}")
    return dataset[split]


def validate_dataset(dataset: Dataset, spec: DatasetSpec, location: str) -> None:
    """校验数据集的字段、行数以及首尾样本是否完整。"""
    missing = sorted(set(spec.required_columns) - set(dataset.column_names))
    if missing:
        raise ValueError(
            f"{spec.repo_id} 在 {location} 缺少字段 {missing}；"
            f"实际字段为 {dataset.column_names}"
        )
    if len(dataset) != spec.expected_rows:
        raise ValueError(
            f"{spec.repo_id} 在 {location} 应有 {spec.expected_rows} 条，"
            f"实际为 {len(dataset)} 条"
        )

    # 首尾抽样既能发现错误 split，也不会为了检查而把 30k 条长解答全部读入内存。
    for index in (0, len(dataset) - 1):
        row = dataset[index]
        empty = [
            column
            for column in spec.required_columns
            if not str(row.get(column, "")).strip()
        ]
        if empty:
            raise ValueError(f"{spec.repo_id} 第 {index} 条存在空字段: {empty}")


def download_one(
    spec: DatasetSpec, data_root: Path, cache_dir: Path, force: bool
) -> None:
    """下载一份固定 revision 的数据集，落盘后再次读取校验。"""
    output_path = data_root / spec.output_name
    if output_path.exists() and not force:
        local = as_single_split(load_from_disk(str(output_path)), spec.split)
        validate_dataset(local, spec, str(output_path))
        print(
            f"[skip] {spec.key}: 本地数据已经存在且校验通过 "
            f"({len(local):,} rows) -> {output_path}"
        )
        return

    if output_path.exists():
        shutil.rmtree(output_path)

    print(f"[download] {spec.repo_id}@{spec.revision} [{spec.split}]")
    dataset = load_dataset(
        spec.repo_id,
        revision=spec.revision,
        split=spec.split,
        cache_dir=str(cache_dir),
    )
    if not isinstance(dataset, Dataset):
        raise TypeError(f"期望 Dataset，实际得到 {type(dataset)!r}")
    validate_dataset(dataset, spec, "Hugging Face")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))

    saved = as_single_split(load_from_disk(str(output_path)), spec.split)
    validate_dataset(saved, spec, str(output_path))
    print(
        f"[done] {spec.key}: {len(saved):,} rows, "
        f"columns={saved.column_names} -> {output_path}"
    )


def main(args: argparse.Namespace) -> None:
    """根据用户选择依次下载 OPSD 训练集和 AIME25 评测集。"""
    data_root = args.data_root.resolve()
    cache_dir = (args.cache_dir or data_root / ".cache").resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    selected = DATASETS.values() if args.only == "all" else (DATASETS[args.only],)
    for spec in selected:
        download_one(spec, data_root, cache_dir, args.force)


if __name__ == "__main__":
    main(parse_args())
