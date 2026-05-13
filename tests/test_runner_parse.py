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
