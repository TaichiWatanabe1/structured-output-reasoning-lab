"""reporting.py の集計関数を合成 JSONL でテスト。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sor_lab.reporting import (
    accuracy_table,
    key_order_warnings,
    majority_accuracy_table,
    order_effect_mcnemar,
    runner_spread,
    write_summary,
)


def _row(
    *,
    condition_id: str,
    runner: str,
    temp_set: str,
    try_idx: int,
    question_id: str,
    gold: int,
    parsed: int | None,
    keys: list[str] | None = None,
    step_count: int = 1,
    model_key: str = "gpt-4.1",
    key_order_violation: bool = False,
) -> dict[str, object]:
    return {
        "run_id": "test",
        "condition_id": condition_id,
        "runner": runner,
        "model_key": model_key,
        "deployment": f"{model_key}-deploy",
        "reasoning_effort": None,
        "model_string": f"{model_key}-2025",
        "temp_set": temp_set,
        "temperature": 0.0 if temp_set == "det" else 0.7,
        "try_idx": try_idx,
        "question_id": question_id,
        "question": "Q",
        "gold_answer": gold,
        "prompt": {"system": "s", "user": "u"},
        "raw_response": json.dumps({"answer": parsed}) if parsed is not None else "",
        "raw_response_keys": keys,
        "parsed_answer": parsed,
        "parsed_reasoning": None,
        "parsed_reasoning_step_count": step_count,
        "parse_failed": parsed is None,
        "correct": parsed == gold if parsed is not None else False,
        "key_order_violation": key_order_violation,
        "latency_ms": 100,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "run_order_idx": 0,
    }


def _make_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    # 4 問。AnswerFirst: 1問正解。ReasoningFirst: 3問正解。Runner = openai_sdk, Model = gpt-4.1。
    for qid in range(4):
        rows.append(
            _row(
                condition_id="so_answer_first",
                runner="openai_sdk",
                temp_set="det",
                try_idx=0,
                question_id=f"q{qid}",
                gold=qid,
                parsed=qid if qid == 0 else qid + 100,
                keys=["answer", "reasoning"],
            )
        )
        rows.append(
            _row(
                condition_id="so_reasoning_first",
                runner="openai_sdk",
                temp_set="det",
                try_idx=0,
                question_id=f"q{qid}",
                gold=qid,
                parsed=qid if qid != 3 else qid + 100,
                keys=["reasoning", "answer"],
            )
        )
    return pd.DataFrame(rows)


def test_accuracy_table_basic() -> None:
    df = _make_df()
    tbl = accuracy_table(df)
    assert not tbl.empty
    af = tbl[
        (tbl["condition_id"] == "so_answer_first") & (tbl["runner"] == "openai_sdk")
    ]
    rf = tbl[
        (tbl["condition_id"] == "so_reasoning_first") & (tbl["runner"] == "openai_sdk")
    ]
    assert af["accuracy"].iloc[0] == 0.25
    assert rf["accuracy"].iloc[0] == 0.75
    # model_key 列があり、内容は gpt-4.1 だけ
    assert set(tbl["model_key"]) == {"gpt-4.1"}


def test_order_effect_mcnemar_runs() -> None:
    df = _make_df()
    tbl = order_effect_mcnemar(df)
    assert not tbl.empty
    row = tbl.iloc[0]
    assert row["answer_first_acc"] == 0.25
    assert row["reasoning_first_acc"] == 0.75
    assert row["model_key"] == "gpt-4.1"
    # q0: AF=correct, RF=correct (both)
    # q1: AF=wrong, RF=correct (c, RF only)
    # q2: AF=wrong, RF=correct (c, RF only)
    # q3: AF=wrong, RF=wrong (both wrong)
    assert row["b_af_only"] == 0
    assert row["c_rf_only"] == 2
    assert row["n_paired"] == 4
    # Newcombe CI が記録されている
    assert "newcombe_lower_pp" in row.index
    assert "newcombe_upper_pp" in row.index
    # 点推定 +50pt なので CI 下限は負, 上限は正くらいのレンジに収まる
    assert row["newcombe_lower_pp"] <= row["diff_pp"] <= row["newcombe_upper_pp"]


def test_order_effect_separates_models() -> None:
    """同じ条件で 2 model 分のデータがあると、それぞれ独立に集計される。"""
    rows = []
    for model in ("gpt-4.1", "gpt-5"):
        for qid in range(4):
            rows.append(
                _row(
                    condition_id="so_answer_first",
                    runner="openai_sdk",
                    temp_set="det",
                    try_idx=0,
                    question_id=f"q{qid}",
                    gold=qid,
                    parsed=qid if qid == 0 else qid + 100,
                    keys=["answer", "reasoning"],
                    model_key=model,
                )
            )
            rows.append(
                _row(
                    condition_id="so_reasoning_first",
                    runner="openai_sdk",
                    temp_set="det",
                    try_idx=0,
                    question_id=f"q{qid}",
                    gold=qid,
                    parsed=qid if qid != 3 else qid + 100,
                    keys=["reasoning", "answer"],
                    model_key=model,
                )
            )
    df = pd.DataFrame(rows)
    tbl = order_effect_mcnemar(df)
    assert len(tbl) == 2
    assert set(tbl["model_key"]) == {"gpt-4.1", "gpt-5"}


def test_runner_spread_single_runner_zero() -> None:
    df = _make_df()
    spread = runner_spread(df)
    # Runner が 1 つしか無い場合は max=min なので 0
    for _, row in spread.iterrows():
        assert row["max_minus_min_pp"] == 0.0


def test_majority_accuracy_with_sc() -> None:
    # sc 試行を 3 つ作って多数決を確認
    rows: list[dict[str, object]] = []
    for q in range(2):
        for try_idx in range(3):
            parsed = q if try_idx <= 1 else q + 100  # 2/3 が正解
            rows.append(
                _row(
                    condition_id="so_reasoning_first",
                    runner="openai_sdk",
                    temp_set="sc",
                    try_idx=try_idx,
                    question_id=f"q{q}",
                    gold=q,
                    parsed=parsed,
                    keys=["reasoning", "answer"],
                )
            )
    df = pd.DataFrame(rows)
    tbl = majority_accuracy_table(df)
    assert not tbl.empty
    assert tbl["accuracy"].iloc[0] == 1.0  # どちらも多数決で正解


def test_order_effect_gate_excludes_high_violation_cell() -> None:
    """SO 両条件いずれかで violation_rate > 1% の cell は H1 検定から除外。"""
    import math

    rows: list[dict[str, object]] = []
    n = 200
    # so_answer_first を 200 件、うち 5 件 (2.5%) を key_order_violation=True にする
    for qid in range(n):
        rows.append(
            _row(
                condition_id="so_answer_first",
                runner="openai_sdk",
                temp_set="det",
                try_idx=0,
                question_id=f"q{qid}",
                gold=qid,
                parsed=qid if qid % 2 == 0 else qid + 100,
                keys=["answer", "reasoning"],
                key_order_violation=(qid < 5),
            )
        )
        rows.append(
            _row(
                condition_id="so_reasoning_first",
                runner="openai_sdk",
                temp_set="det",
                try_idx=0,
                question_id=f"q{qid}",
                gold=qid,
                parsed=qid,
                keys=["reasoning", "answer"],
                key_order_violation=False,
            )
        )
    tbl = order_effect_mcnemar(pd.DataFrame(rows))
    assert not tbl.empty
    row = tbl.iloc[0]
    assert bool(row["excluded_by_key_order_gate"]) is True
    assert math.isnan(float(row["mcnemar_p"]))
    assert math.isnan(float(row["newcombe_lower_pp"]))
    # 点推定 (diff_pp) は除外関係なく出る (記述目的)
    assert row["diff_pp"] > 0


def test_order_effect_gate_passes_when_violation_is_below_threshold() -> None:
    """違反率 0% の cell は通常どおり mcnemar_p / CI が計算される。"""
    import math

    df = _make_df()  # default で violation=False
    tbl = order_effect_mcnemar(df)
    assert not tbl.empty
    row = tbl.iloc[0]
    assert bool(row["excluded_by_key_order_gate"]) is False
    assert not math.isnan(float(row["mcnemar_p"]))


def test_key_order_warnings_uses_violation_flag() -> None:
    df = pd.DataFrame(
        [
            _row(
                condition_id="so_reasoning_first",
                runner="openai_sdk",
                temp_set="det",
                try_idx=0,
                question_id="q0",
                gold=1,
                parsed=1,
                keys=["answer", "reasoning"],
                key_order_violation=True,
            )
        ]
    )
    tbl = key_order_warnings(df)
    assert tbl.iloc[0]["key_order_mismatches"] == 1
    assert tbl.iloc[0]["violation_rate"] == 1.0


def test_write_summary_end_to_end(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "runs" / "test_run"
    run_dir.mkdir(parents=True)
    jsonl = run_dir / "raw.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for row in _make_df().to_dict(orient="records"):
            f.write(json.dumps(row) + "\n")
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "stage": "b1",
                "git_commit": "abc",
                "git_dirty": False,
                "api_version": "2025-04-01-preview",
                "exec_seed": 42,
                "models": [
                    {"key": "gpt-4.1", "deployment": "gpt-4.1-deploy", "reasoning_effort": None},
                ],
                "conditions_yaml_sha256": "x" * 64,
            }
        ),
        encoding="utf-8",
    )
    out = write_summary(run_id="test_run", results_root=tmp_path / "results" / "runs")
    text = out.read_text(encoding="utf-8")
    assert "Summary" in text
    assert "Accuracy by" in text
    assert "so_reasoning_first" in text
    assert "gpt-4.1" in text
