"""experiment.py のコール列構築・シャッフル・key_order_violation 判定の単体テスト。"""

from __future__ import annotations

from sor_lab.config import ModelSpec
from sor_lab.datasets import Problem
from sor_lab.experiment import (
    CallSpec,
    ExperimentConfig,
    StageSpec,
    TempSet,
    _key_order_violation,
    build_call_list,
    shuffle_call_list,
)


def _problems(n: int) -> list[Problem]:
    return [
        Problem(question_id=f"mgsm_ja_{i}", question=f"q{i}", gold_answer=i, reference="")
        for i in range(n)
    ]


def _cfg() -> ExperimentConfig:
    return ExperimentConfig(
        conditions=["plain_direct", "so_reasoning_first"],
        runners=["openai_sdk"],
        temp_sets={
            "det": TempSet("det", 0.0, 1),
            "sc": TempSet("sc", 0.7, 3),
        },
        dataset_seed=42,
        dataset_n=2,
        stages={},
        raw_yaml="",
    )


def _stage() -> StageSpec:
    return StageSpec(
        name="test",
        conditions=["plain_direct", "so_reasoning_first"],
        runners=["openai_sdk"],
        temp_sets=["det", "sc"],
        n_questions=None,
    )


def _models() -> list[ModelSpec]:
    return [
        ModelSpec("gpt-4.1", "dep-41", None),
        ModelSpec("gpt-5", "dep-5", "minimal"),
    ]


def test_build_call_list_full_product() -> None:
    plan = build_call_list(_cfg(), _stage(), _models(), _problems(2))
    # 2 cond × 1 runner × (1 det + 3 sc) × 2 model × 2 question = 32
    assert len(plan) == 2 * 1 * (1 + 3) * 2 * 2


def test_build_call_list_callspec_fields() -> None:
    plan = build_call_list(_cfg(), _stage(), _models(), _problems(1))
    for c in plan:
        assert isinstance(c, CallSpec)
        assert c.condition_id in {"plain_direct", "so_reasoning_first"}
        assert c.runner_name == "openai_sdk"
        assert c.model_key in {"gpt-4.1", "gpt-5"}
        assert c.question_id == "mgsm_ja_0"


def test_shuffle_call_list_is_deterministic() -> None:
    plan = build_call_list(_cfg(), _stage(), _models(), _problems(2))
    a = shuffle_call_list(plan, seed=42)
    b = shuffle_call_list(plan, seed=42)
    assert a == b


def test_shuffle_call_list_changes_order_but_preserves_set() -> None:
    plan = build_call_list(_cfg(), _stage(), _models(), _problems(2))
    shuffled = shuffle_call_list(plan, seed=42)
    assert set(shuffled) == set(plan)
    # 32 個もあれば順序は必ず変わる
    assert shuffled != plan


def test_key_order_violation_plain_is_false() -> None:
    """Plain 系条件は対象外なので常に False。"""
    assert _key_order_violation("plain_direct", ["answer"]) is False
    assert _key_order_violation("plain_direct", None) is False


def test_key_order_violation_so_reasoning_first_correct_order() -> None:
    """so_reasoning_first の declared order は [reasoning, answer]。一致すれば False。"""
    assert _key_order_violation("so_reasoning_first", ["reasoning", "answer"]) is False


def test_key_order_violation_so_reasoning_first_wrong_order() -> None:
    """逆順なら True。"""
    assert _key_order_violation("so_reasoning_first", ["answer", "reasoning"]) is True


def test_key_order_violation_parse_failed_keys_none() -> None:
    """raw_response_keys=None (parse 失敗) は parse_failed 側で扱うので False。"""
    assert _key_order_violation("so_answer_first", None) is False
