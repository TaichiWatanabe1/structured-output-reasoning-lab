"""データセットサンプリングのテスト (HuggingFace 接続なしで動く部分のみ)。"""

from __future__ import annotations

from sor_lab.datasets import MGSM_JA_FULL_SIZE, sample_subset


def _make_rows(n: int) -> list[dict[str, object]]:
    """ダミー MGSM-ja 行を生成 (`question`, `answer`, `answer_number`)。"""
    return [
        {
            "question": f"問題 {i}: {i} + {i} はいくつですか？",
            "answer": f"説明: {i} に {i} を足すと {i * 2} です。",
            "answer_number": i * 2,
        }
        for i in range(n)
    ]


def test_sample_subset_size_smaller_than_full() -> None:
    rows = _make_rows(250)
    subset = sample_subset(rows, n=50, seed=42)
    assert len(subset) == 50


def test_sample_subset_full_returns_all_sorted() -> None:
    rows = _make_rows(MGSM_JA_FULL_SIZE)
    subset = sample_subset(rows, n=MGSM_JA_FULL_SIZE, seed=42)
    assert len(subset) == MGSM_JA_FULL_SIZE
    indices = [int(p.question_id.removeprefix("mgsm_ja_")) for p in subset]
    assert indices == sorted(indices)
    assert indices == list(range(MGSM_JA_FULL_SIZE))


def test_sample_subset_seed_reproducible() -> None:
    rows = _make_rows(250)
    a = sample_subset(rows, n=50, seed=42)
    b = sample_subset(rows, n=50, seed=42)
    assert [p.question_id for p in a] == [p.question_id for p in b]


def test_sample_subset_different_seed_differs() -> None:
    rows = _make_rows(250)
    a = sample_subset(rows, n=50, seed=42)
    b = sample_subset(rows, n=50, seed=43)
    assert [p.question_id for p in a] != [p.question_id for p in b]


def test_sample_subset_question_id_format() -> None:
    rows = _make_rows(250)
    subset = sample_subset(rows, n=10, seed=42)
    for p in subset:
        assert p.question_id.startswith("mgsm_ja_")
        idx = int(p.question_id.removeprefix("mgsm_ja_"))
        assert 0 <= idx < 250
        assert p.gold_answer == idx * 2


def test_sample_subset_sorted_by_index() -> None:
    rows = _make_rows(250)
    subset = sample_subset(rows, n=50, seed=42)
    indices = [int(p.question_id.removeprefix("mgsm_ja_")) for p in subset]
    assert indices == sorted(indices)
