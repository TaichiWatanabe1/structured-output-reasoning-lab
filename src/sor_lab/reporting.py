"""JSONL → summary.md 集計。

主要指標:
  - 条件 × Runner × temp_set の正解率 + Wilson 95% CI
  - temp_set=sc は多数決の正解率も別計算
  - 順序効果 (so_reasoning_first vs so_answer_first) を McNemar 検定
  - Runner 差分 (3 Runner の max-min)
  - キー順 warning 集計
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from sor_lab.evaluation import majority_vote, wilson_ci


def load_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---- 集計 -------------------------------------------------------------------


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """`(condition_id, runner, temp_set)` ごとの正解率と Wilson CI、parse 失敗率。

    temp_set=sc は単発平均 (try ごとの correct を平均) のみ。
    多数決は `majority_accuracy_table` で別途。
    """
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(["condition_id", "runner", "temp_set"])

    records: list[dict[str, Any]] = []
    for (cond, runner, ts), g in grouped:
        n = len(g)
        successes = int(g["correct"].sum())
        ci = wilson_ci(successes, n)
        records.append(
            {
                "condition_id": cond,
                "runner": runner,
                "temp_set": ts,
                "n": n,
                "accuracy": successes / n if n else 0.0,
                "ci_lower": ci.lower,
                "ci_upper": ci.upper,
                "parse_fail_rate": g["parse_failed"].mean() if "parse_failed" in g else 0.0,
                "mean_step_count": (
                    g["parsed_reasoning_step_count"].mean()
                    if "parsed_reasoning_step_count" in g
                    else 0.0
                ),
                "mean_latency_ms": g["latency_ms"].mean() if "latency_ms" in g else 0.0,
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["temp_set", "condition_id", "runner"]
    )


def majority_accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """temp_set=sc の多数決正解率。試行を集約して 1 question 1 answer にする。"""
    if df.empty:
        return pd.DataFrame()
    sc = df[df["temp_set"] == "sc"]
    if sc.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for (cond, runner), g in sc.groupby(["condition_id", "runner"]):
        per_question = g.groupby("question_id").agg(
            answers=("parsed_answer", list),
            gold=("gold_answer", "first"),
        )
        per_question["majority"] = per_question["answers"].apply(majority_vote)
        per_question["correct_majority"] = per_question["majority"] == per_question["gold"]
        n = len(per_question)
        successes = int(per_question["correct_majority"].sum())
        ci = wilson_ci(successes, n)
        records.append(
            {
                "condition_id": cond,
                "runner": runner,
                "temp_set": "sc_majority",
                "n": n,
                "accuracy": successes / n if n else 0.0,
                "ci_lower": ci.lower,
                "ci_upper": ci.upper,
            }
        )
    return pd.DataFrame.from_records(records).sort_values(["condition_id", "runner"])


def order_effect_mcnemar(df: pd.DataFrame) -> pd.DataFrame:
    """順序効果 H1: so_reasoning_first vs so_answer_first を Runner ごとに検定。

    同じ question_id 上のペア比較。temp_set=det のみを対象。
    """
    if df.empty:
        return pd.DataFrame()
    from statsmodels.stats.contingency_tables import mcnemar

    det = df[df["temp_set"] == "det"]
    af = det[det["condition_id"] == "so_answer_first"]
    rf = det[det["condition_id"] == "so_reasoning_first"]
    if af.empty or rf.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for runner in sorted(set(af["runner"]) & set(rf["runner"])):
        af_r = af[af["runner"] == runner].set_index("question_id")["correct"]
        rf_r = rf[rf["runner"] == runner].set_index("question_id")["correct"]
        common = af_r.index.intersection(rf_r.index)
        if len(common) == 0:
            continue
        af_c = af_r.loc[common].astype(bool)
        rf_c = rf_r.loc[common].astype(bool)
        # contingency:
        # b = af correct, rf wrong  (RF が劣る方向)
        # c = af wrong, rf correct  (RF が優る方向)
        b = int(((af_c) & (~rf_c)).sum())
        c = int(((~af_c) & (rf_c)).sum())
        # 2x2 table for mcnemar
        table = [[int((af_c & rf_c).sum()), b], [c, int((~af_c & ~rf_c).sum())]]
        try:
            result = mcnemar(table, exact=True)
            pval = float(result.pvalue)
        except Exception:  # noqa: BLE001
            pval = float("nan")
        af_acc = float(af_c.mean())
        rf_acc = float(rf_c.mean())
        records.append(
            {
                "runner": runner,
                "n_paired": int(len(common)),
                "answer_first_acc": af_acc,
                "reasoning_first_acc": rf_acc,
                "diff_pp": (rf_acc - af_acc) * 100.0,
                "b_af_only": b,
                "c_rf_only": c,
                "mcnemar_p": pval,
            }
        )
    return pd.DataFrame.from_records(records)


def runner_spread(df: pd.DataFrame) -> pd.DataFrame:
    """H4: 同条件で 3 Runner の正解率 max-min を計算。"""
    if df.empty:
        return pd.DataFrame()
    det = df[df["temp_set"] == "det"]
    if det.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for cond, g in det.groupby("condition_id"):
        per_runner = g.groupby("runner")["correct"].mean()
        records.append(
            {
                "condition_id": cond,
                **{f"acc_{r}": float(per_runner.get(r, float("nan"))) for r in per_runner.index},
                "max_minus_min_pp": (per_runner.max() - per_runner.min()) * 100.0,
            }
        )
    return pd.DataFrame.from_records(records)


def key_order_warnings(df: pd.DataFrame) -> pd.DataFrame:
    """raw_response_keys が宣言順と一致しない件数 (Structured 条件のみ)。"""
    if df.empty or "raw_response_keys" not in df.columns:
        return pd.DataFrame()

    from sor_lab.schemas import field_order, get_schema

    expected_cache: dict[str, list[str] | None] = {}

    def expected(cond_id: str) -> list[str] | None:
        if cond_id in expected_cache:
            return expected_cache[cond_id]
        schema_cls = get_schema(cond_id)
        if schema_cls is None:
            expected_cache[cond_id] = None
            return None
        expected_cache[cond_id] = field_order(schema_cls)
        return expected_cache[cond_id]

    records: list[dict[str, Any]] = []
    for (cond, runner), g in df.groupby(["condition_id", "runner"]):
        exp = expected(cond)
        if exp is None:
            continue  # Plain 条件は Structured 検査対象外
        total = 0
        mismatches = 0
        for keys in g["raw_response_keys"]:
            if keys is None:
                continue
            total += 1
            if list(keys) != exp:
                mismatches += 1
        records.append(
            {
                "condition_id": cond,
                "runner": runner,
                "structured_calls": total,
                "key_order_mismatches": mismatches,
            }
        )
    return pd.DataFrame.from_records(records)


# ---- summary.md 生成 --------------------------------------------------------


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no data)_\n"
    return (df.to_markdown(index=False, floatfmt=".4f") or "") + "\n"


def write_summary(
    run_id: str,
    results_root: Path | str = "results/runs",
) -> Path:
    root = Path(results_root) / run_id
    jsonl = root / "raw.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"raw.jsonl not found: {jsonl}")
    df = load_jsonl(jsonl)

    sections: list[str] = []
    sections.append(f"# Summary — `{run_id}`\n")
    meta_path = root / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sections.append(
            f"- stage: `{meta.get('stage')}`\n"
            f"- git_commit: `{meta.get('git_commit')}` (dirty={meta.get('git_dirty')})\n"
            f"- deployment: `{meta.get('deployment')}` / api_version: `{meta.get('api_version')}`\n"
            f"- conditions.yaml sha256: `{meta.get('conditions_yaml_sha256')}`\n"
        )

    sections.append("\n## Accuracy by (condition, runner, temp_set)\n")
    sections.append(_df_to_markdown(accuracy_table(df)))

    sections.append("\n## Self-consistency majority vote accuracy (temp_set=sc)\n")
    sections.append(_df_to_markdown(majority_accuracy_table(df)))

    sections.append("\n## H1 — Order effect (so_reasoning_first vs so_answer_first)\n")
    sections.append(_df_to_markdown(order_effect_mcnemar(df)))

    sections.append("\n## H4 — Runner spread (max - min, percentage points)\n")
    sections.append(_df_to_markdown(runner_spread(df)))

    sections.append("\n## Key order warnings (raw_response_keys vs declared)\n")
    sections.append(_df_to_markdown(key_order_warnings(df)))

    out = root / "summary.md"
    out.write_text("".join(sections), encoding="utf-8")
    return out


__all__ = [
    "load_jsonl",
    "accuracy_table",
    "majority_accuracy_table",
    "order_effect_mcnemar",
    "runner_spread",
    "key_order_warnings",
    "write_summary",
]
