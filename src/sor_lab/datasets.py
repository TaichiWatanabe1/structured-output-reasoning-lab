"""MGSM-ja サブセットのロード。

`datasets` ライブラリで `juletxara/mgsm` の `ja` split を全件 (250 問) ロードし、
そのまま `Problem` に変換する。本実験は MGSM 全件投入 (n=250) を前提とする。

任意の `n` < 250 を指定した場合は `random.Random(seed).sample(...)` で
シード固定サブセットを取る (smoke test 等の小規模実行向け)。
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any


DEFAULT_SEED = 42
MGSM_JA_FULL_SIZE = 250
DEFAULT_SUBSET_SIZE = MGSM_JA_FULL_SIZE


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


def load_mgsm_ja(cache_dir: str | None = None) -> list[dict[str, Any]]:
    """MGSM `ja` split を全件 (250) ロードして dict のリストとして返す。"""
    _maybe_set_cache_dir()
    # `datasets` は重い import なので関数内で遅延 import する。
    from datasets import load_dataset

    ds = load_dataset("juletxara/mgsm", "ja", split="test", cache_dir=cache_dir)
    return [dict(row) for row in ds]


def _row_to_problem(idx: int, row: dict[str, Any]) -> Problem:
    """MGSM 1 行を `Problem` に変換。

    MGSM スキーマ: `question` (str), `answer_number` (int), `answer` (str, 解説本文)。
    """
    answer_number = row.get("answer_number")
    if answer_number is None:
        raise ValueError(f"MGSM row {idx} has no answer_number: {row!r}")
    return Problem(
        question_id=f"mgsm_ja_{idx}",
        question=str(row["question"]),
        gold_answer=int(answer_number),
        reference=str(row.get("answer", "")),
    )


def sample_subset(
    rows: list[dict[str, Any]],
    n: int = DEFAULT_SUBSET_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[Problem]:
    """シード固定サブセット。`question_id` は `mgsm_ja_<元 index>` 形式。

    `n` が `len(rows)` 以上ならソート済みの全件を返す (シャッフルしない)。
    `n` < `len(rows)` の場合は `random.Random(seed).sample(...)` で抽出。
    """
    total = len(rows)
    if n >= total:
        chosen = list(range(total))
    else:
        chosen = random.Random(seed).sample(range(total), n)
        chosen.sort()

    return [_row_to_problem(idx, rows[idx]) for idx in chosen]


def load_subset(
    n: int = DEFAULT_SUBSET_SIZE,
    seed: int = DEFAULT_SEED,
    cache_dir: str | None = None,
) -> list[Problem]:
    """全体をロードしてサブセットを取る便利関数。デフォルトは MGSM-ja 全 250 問。"""
    rows = load_mgsm_ja(cache_dir=cache_dir)
    return sample_subset(rows, n=n, seed=seed)


__all__ = [
    "Problem",
    "DEFAULT_SEED",
    "DEFAULT_SUBSET_SIZE",
    "MGSM_JA_FULL_SIZE",
    "load_mgsm_ja",
    "sample_subset",
    "load_subset",
]
