"""評価ロジック (答え抽出 / ステップ数 / 統計指標) のテスト。"""

from __future__ import annotations

import math

import pytest

from sor_lab.evaluation import (
    count_reasoning_steps,
    majority_vote,
    parse_gold_answer,
    parse_plain_answer,
    wilson_ci,
)


# ---- gold answer ----


@pytest.mark.parametrize(
    "ref,expected",
    [
        ("Step 1...\n#### 18", 18),
        ("complex calculation\n#### 1,200", 1200),
        ("#### -5", -5),
        ("foo\n#### 3.0", 3),
    ],
)
def test_parse_gold_answer(ref: str, expected: int) -> None:
    assert parse_gold_answer(ref) == expected


def test_parse_gold_answer_missing_marker() -> None:
    with pytest.raises(ValueError):
        parse_gold_answer("no marker")


# ---- plain answer ----


@pytest.mark.parametrize(
    "text,expected",
    [
        ("42", 42),
        ("The result is 42.", 42),
        ("Step by step...\nAnswer: 42", 42),
        ("Step by step...\nAnswer: 1,200", 1200),
        ("Final: -7\nAnswer: -7", -7),
        ("Answer: $1,200", 1200),
        ("Answer: 42.0", 42),
        ("answer:  100  ", 100),
    ],
)
def test_parse_plain_answer_success(text: str, expected: int) -> None:
    assert parse_plain_answer(text) == expected


@pytest.mark.parametrize("text", ["", "no number here", "Answer: not-a-number"])
def test_parse_plain_answer_failure_returns_none(text: str) -> None:
    assert parse_plain_answer(text) is None


def test_parse_plain_answer_prefers_answer_prefix_over_trailing() -> None:
    # 末尾は "100" だが "Answer: 42" を優先
    text = "Some text 100. More text. Answer: 42 done."
    assert parse_plain_answer(text) == 42


# ---- step count ----


def test_count_reasoning_steps_list() -> None:
    assert count_reasoning_steps(["a", "b", "c"]) == 3
    assert count_reasoning_steps([]) == 0


def test_count_reasoning_steps_none() -> None:
    assert count_reasoning_steps(None) == 0


def test_count_reasoning_steps_newlines() -> None:
    text = "step one\nstep two\nstep three"
    assert count_reasoning_steps(text) == 3


def test_count_reasoning_steps_numbered() -> None:
    text = "1. first\n2. second\n3. third"
    assert count_reasoning_steps(text) == 3


def test_count_reasoning_steps_numbered_prefers_explicit() -> None:
    # 改行 4 行だが番号付きは 2 つ、改行に fallback (numbered < 2 件)
    text = "preamble\n1. first\nfollowup"
    assert count_reasoning_steps(text) == 3


def test_count_reasoning_steps_blank_lines_ignored() -> None:
    text = "\n\nstep one\n\nstep two\n\n"
    assert count_reasoning_steps(text) == 2


# ---- wilson CI ----


def test_wilson_ci_zero_n() -> None:
    ci = wilson_ci(0, 0)
    assert ci.lower == 0.0
    assert ci.upper == 1.0


def test_wilson_ci_50_50() -> None:
    ci = wilson_ci(25, 50)
    # 50% は対称、おおよそ [0.36, 0.64]
    assert 0.35 < ci.lower < 0.40
    assert 0.60 < ci.upper < 0.65


def test_wilson_ci_perfect() -> None:
    ci = wilson_ci(50, 50)
    assert math.isclose(ci.upper, 1.0, abs_tol=1e-9)
    assert ci.lower > 0.9


def test_wilson_ci_zero_successes() -> None:
    ci = wilson_ci(0, 50)
    assert math.isclose(ci.lower, 0.0, abs_tol=1e-9)
    assert ci.upper < 0.1


# ---- majority vote ----


def test_majority_vote_basic() -> None:
    assert majority_vote([1, 2, 2, 3, 2]) == 2


def test_majority_vote_tie_returns_min() -> None:
    assert majority_vote([1, 2, 1, 2]) == 1


def test_majority_vote_filters_none() -> None:
    assert majority_vote([None, 5, None, 5, 7]) == 5


def test_majority_vote_all_none() -> None:
    assert majority_vote([None, None]) is None


def test_majority_vote_empty() -> None:
    assert majority_vote([]) is None
