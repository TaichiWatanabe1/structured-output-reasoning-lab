"""Runner 共通の `_parse_response` を network なしで検証。"""

from __future__ import annotations

import json

from sor_lab.runners.openai_sdk import _parse_response


def test_parse_plain_text_success() -> None:
    r = _parse_response(
        raw="The answer is 42.\nAnswer: 42",
        structured=False,
        latency_ms=100,
        prompt_tokens=10,
        completion_tokens=5,
    )
    assert r.parsed_answer == 42
    assert r.parse_failed is False
    assert r.raw_response_keys is None
    assert r.parsed_reasoning == "The answer is 42.\nAnswer: 42"


def test_parse_plain_text_failure() -> None:
    r = _parse_response(
        raw="no number here",
        structured=False,
        latency_ms=100,
        prompt_tokens=None,
        completion_tokens=None,
    )
    assert r.parsed_answer is None
    assert r.parse_failed is True


def test_parse_structured_reasoning_first() -> None:
    raw = json.dumps({"reasoning": "step1; step2", "answer": 7})
    r = _parse_response(
        raw=raw, structured=True, latency_ms=200, prompt_tokens=20, completion_tokens=10
    )
    assert r.parsed_answer == 7
    assert r.parsed_reasoning == "step1; step2"
    assert r.raw_response_keys == ["reasoning", "answer"]
    assert r.parse_failed is False


def test_parse_structured_answer_first() -> None:
    raw = json.dumps({"answer": 7, "reasoning": "step1"})
    r = _parse_response(
        raw=raw, structured=True, latency_ms=200, prompt_tokens=20, completion_tokens=10
    )
    assert r.parsed_answer == 7
    assert r.raw_response_keys == ["answer", "reasoning"]


def test_parse_structured_step_list() -> None:
    raw = json.dumps({"reasoning_steps": ["a", "b", "c"], "answer": 5})
    r = _parse_response(
        raw=raw, structured=True, latency_ms=100, prompt_tokens=10, completion_tokens=5
    )
    assert r.parsed_answer == 5
    assert r.parsed_reasoning == ["a", "b", "c"]
    assert r.raw_response_keys == ["reasoning_steps", "answer"]


def test_parse_structured_invalid_json() -> None:
    r = _parse_response(
        raw="not json {", structured=True, latency_ms=50, prompt_tokens=None,
        completion_tokens=None,
    )
    assert r.parsed_answer is None
    assert r.raw_response_keys is None
    assert r.parse_failed is True


def test_parse_structured_missing_answer() -> None:
    raw = json.dumps({"reasoning": "step1"})
    r = _parse_response(
        raw=raw, structured=True, latency_ms=50, prompt_tokens=None,
        completion_tokens=None,
    )
    assert r.parsed_answer is None
    assert r.parse_failed is True
    # キー順は読めるはず (JSON は valid)
    assert r.raw_response_keys == ["reasoning"]


def test_parse_structured_with_schema_extra_key_fails() -> None:
    """`extra="forbid"` のスキーマで余分なキーがあれば ValidationError → parse_failed。"""
    from sor_lab.schemas import AnswerFirst

    raw = json.dumps({"answer": 7, "reasoning": "x", "foo": 1})
    r = _parse_response(
        raw=raw,
        structured=True,
        latency_ms=10,
        prompt_tokens=None,
        completion_tokens=None,
        schema_cls=AnswerFirst,
    )
    assert r.parse_failed is True
    assert r.parsed_answer is None
    # raw_response_keys は検証前に読まれるので残る (モデル生成順の証跡)
    assert r.raw_response_keys == ["answer", "reasoning", "foo"]


def test_parse_structured_with_schema_missing_reasoning_fails() -> None:
    """`reasoning` が無いと ValidationError → parse_failed (`answer` だけでは通さない)。"""
    from sor_lab.schemas import AnswerFirst

    raw = json.dumps({"answer": 7})
    r = _parse_response(
        raw=raw,
        structured=True,
        latency_ms=10,
        prompt_tokens=None,
        completion_tokens=None,
        schema_cls=AnswerFirst,
    )
    assert r.parse_failed is True
    assert r.parsed_answer is None


def test_parse_structured_step_constrained_max_length_violation() -> None:
    """`reasoning_steps` の長さが `max_length` を超えれば parse_failed。"""
    from sor_lab.schemas import make_step_constrained

    schema_cls = make_step_constrained(n_steps=3, answer_first=False)
    raw = json.dumps({"reasoning_steps": ["a", "b", "c", "d"], "answer": 5})
    r = _parse_response(
        raw=raw,
        structured=True,
        latency_ms=10,
        prompt_tokens=None,
        completion_tokens=None,
        schema_cls=schema_cls,
    )
    assert r.parse_failed is True
    assert r.parsed_answer is None


def test_parse_structured_with_schema_valid() -> None:
    """正規の応答は parse 成功し、raw_response_keys がモデル生成順を保つ。"""
    from sor_lab.schemas import ReasoningFirst

    raw = json.dumps({"reasoning": "step1; step2", "answer": 42})
    r = _parse_response(
        raw=raw,
        structured=True,
        latency_ms=10,
        prompt_tokens=None,
        completion_tokens=None,
        schema_cls=ReasoningFirst,
    )
    assert r.parse_failed is False
    assert r.parsed_answer == 42
    assert r.parsed_reasoning == "step1; step2"
    assert r.raw_response_keys == ["reasoning", "answer"]


def test_parse_structured_with_schema_wrong_key_order_still_validates() -> None:
    """key 順が宣言と異なっても、内容が valid なら parse は成功する
    (key 順は `key_order_violation` 別途集計の対象。`_parse_response` では関与しない)。"""
    from sor_lab.schemas import ReasoningFirst

    raw = json.dumps({"answer": 42, "reasoning": "step1"})
    r = _parse_response(
        raw=raw,
        structured=True,
        latency_ms=10,
        prompt_tokens=None,
        completion_tokens=None,
        schema_cls=ReasoningFirst,
    )
    assert r.parse_failed is False
    assert r.parsed_answer == 42
    # raw_response_keys は検証前の dict.keys() (= モデル実生成順)
    assert r.raw_response_keys == ["answer", "reasoning"]
