"""プロンプト定数の意図しない揺れを検出する (日本語固定)。"""

from __future__ import annotations

import hashlib

import pytest

from sor_lab.prompts import PROMPTS, get_prompt


def test_all_five_conditions_present() -> None:
    assert set(PROMPTS.keys()) == {
        "plain_direct",
        "plain_cot",
        "so_answer_first",
        "so_reasoning_first",
        "so_step_constrained",
    }


def test_so_conditions_share_system_prompt() -> None:
    """Structured 系 3 条件は system プロンプトが完全一致 (順序のみが独立変数)。"""
    so_systems = {
        PROMPTS["so_answer_first"].system,
        PROMPTS["so_reasoning_first"].system,
        PROMPTS["so_step_constrained"].system,
    }
    assert len(so_systems) == 1


def test_plain_cot_contains_step_marker_ja() -> None:
    assert "順を追って" in PROMPTS["plain_cot"].system
    assert "Answer:" in PROMPTS["plain_cot"].system


def test_plain_direct_restricts_to_integer_only_ja() -> None:
    sys_prompt = PROMPTS["plain_direct"].system
    assert "整数" in sys_prompt
    assert "他の文字" in sys_prompt or "だけ" in sys_prompt


def test_user_template_renders_question() -> None:
    s, u = PROMPTS["plain_direct"].render("1 + 1 はいくつですか？")
    assert "1 + 1 はいくつですか？" in u
    assert "{question}" not in u
    assert s == PROMPTS["plain_direct"].system


def test_get_prompt_handles_derived_ids() -> None:
    p = get_prompt("so_step_constrained:n_steps=5:answer_first=False")
    assert p is PROMPTS["so_step_constrained"]


def test_get_prompt_unknown_id() -> None:
    with pytest.raises(KeyError):
        get_prompt("nonexistent_condition")


def test_prompts_hashes_are_stable() -> None:
    """プロンプト全体を 1 つのダイジェストに固めて、不注意な変更を検出する。
    意図的にプロンプトを変えた場合は、本テストの期待値も同 PR で更新すること。
    """
    canonical = "\n--\n".join(
        f"{k}\n{v.system}\n{v.user}" for k, v in sorted(PROMPTS.items())
    )
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    expected = "8243642ca5d9161e9112da54f8f35e5d3a2d14d0ac94b15ff21c6f247ec450cd"
    assert actual == expected, (
        f"prompts.py changed. update expected digest if intentional. got={actual}"
    )
