"""GSM8K サブセットのロード。

`datasets` ライブラリで `gsm8k` (`main` config) の `test` split をロードし、
`random.Random(42).sample(...)` で seed 固定のサブセットを抽出する。
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

from sor_lab.evaluation import parse_gold_answer


DEFAULT_SEED = 42
DEFAULT_SUBSET_SIZE = 50


@dataclass(frozen=True)
class Problem:
    question_id: str
    question: str
    gold_answer: int
    reference: str


def _maybe_set_cache_dir() -> None:
    """`HF_DATASETS_CACHE` が設定済みでなければ `./data/hf_cache` に向ける。"""
    if not os.environ.get("HF_DATASETS_CACHE"):
        os.environ["HF_DATASETS_CACHE"] = os.path.join(os.getcwd(), "data", "hf_cache")


def load_gsm8k_test(cache_dir: str | None = None) -> list[dict[str, Any]]:
    """GSM8K `main/test` を全件ロードして dict のリストとして返す。"""
    _maybe_set_cache_dir()
    # `datasets` は重い import なので関数内で遅延 import する。
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split="test", cache_dir=cache_dir)
    return [dict(row) for row in ds]


def sample_subset(
    rows: list[dict[str, Any]],
    n: int = DEFAULT_SUBSET_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[Problem]:
    """シード固定サブセット。`question_id` は `gsm8k_test_<元 index>` 形式。

    `random.Random(seed).sample(...)` の挙動が Python パッチバージョンで変わると
    再現性が壊れるため、`requires-python==3.12.4` で固定している前提。
    """
    indices = list(range(len(rows)))
    chosen = random.Random(seed).sample(indices, n)
    chosen.sort()

    out: list[Problem] = []
    for idx in chosen:
        row = rows[idx]
        reference = str(row["answer"])
        out.append(
            Problem(
                question_id=f"gsm8k_test_{idx}",
                question=str(row["question"]),
                gold_answer=parse_gold_answer(reference),
                reference=reference,
            )
        )
    return out


def load_subset(
    n: int = DEFAULT_SUBSET_SIZE,
    seed: int = DEFAULT_SEED,
    cache_dir: str | None = None,
) -> list[Problem]:
    """全体をロードして seed 固定サブセットを取る便利関数。"""
    rows = load_gsm8k_test(cache_dir=cache_dir)
    return sample_subset(rows, n=n, seed=seed)


__all__ = [
    "Problem",
    "DEFAULT_SEED",
    "DEFAULT_SUBSET_SIZE",
    "load_gsm8k_test",
    "sample_subset",
    "load_subset",
]
