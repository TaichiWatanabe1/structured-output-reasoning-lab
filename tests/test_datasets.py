"""データセットサンプリングのテスト (HuggingFace 接続なしで動く部分のみ)。"""

from __future__ import annotations

from sor_lab.datasets import sample_subset


def _make_rows(n: int) -> list[dict[str, object]]:
    """ダミー行を生成。answer は `#### <idx>` 形式。"""
    return [
        {
            "question": f"Question number {i}: what is {i} + {i}?",
            "answer": f"Some explanation step.\n#### {i * 2}",
        }
        for i in range(n)
    ]


def test_sample_subset_size() -> None:
    rows = _make_rows(1000)
    subset = sample_subset(rows, n=50, seed=42)
    assert len(subset) == 50


def test_sample_subset_seed_reproducible() -> None:
    rows = _make_rows(1000)
    a = sample_subset(rows, n=50, seed=42)
    b = sample_subset(rows, n=50, seed=42)
    assert [p.question_id for p in a] == [p.question_id for p in b]


def test_sample_subset_different_seed_differs() -> None:
    rows = _make_rows(1000)
    a = sample_subset(rows, n=50, seed=42)
    b = sample_subset(rows, n=50, seed=43)
    assert [p.question_id for p in a] != [p.question_id for p in b]


def test_sample_subset_question_id_format() -> None:
    rows = _make_rows(1000)
    subset = sample_subset(rows, n=10, seed=42)
    for p in subset:
        assert p.question_id.startswith("gsm8k_test_")
        idx = int(p.question_id.removeprefix("gsm8k_test_"))
        assert 0 <= idx < 1000
        assert p.gold_answer == idx * 2


def test_sample_subset_sorted_by_index() -> None:
    rows = _make_rows(1000)
    subset = sample_subset(rows, n=50, seed=42)
    indices = [int(p.question_id.removeprefix("gsm8k_test_")) for p in subset]
    assert indices == sorted(indices)
