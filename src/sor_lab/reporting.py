"""JSONL → summary.md 集計。

主要指標:
  - 条件 × Runner × Model × temp_set の正解率 + Wilson 95% CI (単一比率の表示用)
  - temp_set=sc は多数決の正解率も別計算
  - 順序効果 (so_reasoning_first vs so_answer_first) を McNemar 検定 + Newcombe paired CI
  - Runner 差分 (3 Runner の max-min)
  - キー順序違反 (`key_order_violation` フラグ) を model_key × 条件 × runner で集計
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from sor_lab.evaluation import majority_vote, wilson_ci
from sor_lab.stats import newcombe_paired_diff_ci


# JSONL に model_key が無い旧データ用フォールバック。新実装では必ず付与される。
_DEFAULT_MODEL_KEY = "unknown"


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
    df = pd.DataFrame(rows)
    if "model_key" not in df.columns:
        df["model_key"] = _DEFAULT_MODEL_KEY
    return df


# ---- 集計 -------------------------------------------------------------------


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """`(condition_id, runner, model_key, temp_set)` ごとの正解率と Wilson CI、parse 失敗率。

    Wilson CI は単一比率の点推定表示用。paired 比較は別関数で Newcombe CI を使う。
    temp_set=sc は単発平均 (try ごとの correct を平均) のみ。多数決は `majority_accuracy_table` で別途。
    """
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(["condition_id", "runner", "model_key", "temp_set"])

    records: list[dict[str, Any]] = []
    for (cond, runner, model_key, ts), g in grouped:
        n = len(g)
        successes = int(g["correct"].sum())
        ci = wilson_ci(successes, n)
        records.append(
            {
                "condition_id": cond,
                "runner": runner,
                "model_key": model_key,
                "temp_set": ts,
                "n": n,
                "accuracy": successes / n if n else 0.0,
                "ci_lower": ci.lower,
                "ci_upper": ci.upper,
                "parse_fail_rate": g["parse_failed"].mean() if "parse_failed" in g else 0.0,
                "key_order_violation_rate": (
                    g["key_order_violation"].mean()
                    if "key_order_violation" in g
                    else 0.0
                ),
                "mean_step_count": (
                    g["parsed_reasoning_step_count"].mean()
                    if "parsed_reasoning_step_count" in g
                    else 0.0
                ),
                "mean_latency_ms": g["latency_ms"].mean() if "latency_ms" in g else 0.0,
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["temp_set", "model_key", "condition_id", "runner"]
    )


def majority_accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """temp_set=sc の多数決正解率。試行を集約して 1 question 1 answer にする。"""
    if df.empty:
        return pd.DataFrame()
    sc = df[df["temp_set"] == "sc"]
    if sc.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for (cond, runner, model_key), g in sc.groupby(["condition_id", "runner", "model_key"]):
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
                "model_key": model_key,
                "temp_set": "sc_majority",
                "n": n,
                "accuracy": successes / n if n else 0.0,
                "ci_lower": ci.lower,
                "ci_upper": ci.upper,
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["model_key", "condition_id", "runner"]
    )


_KEY_ORDER_GATE_THRESHOLD = 0.01


def _key_order_violation_rate(g: pd.DataFrame) -> float:
    if "key_order_violation" not in g.columns or g.empty:
        return 0.0
    return float(g["key_order_violation"].fillna(False).astype(bool).mean())


def order_effect_mcnemar(df: pd.DataFrame) -> pd.DataFrame:
    """順序効果 H1: so_reasoning_first vs so_answer_first を Runner × Model ごとに検定。

    同じ question_id 上のペア比較。temp_set=det のみを対象。
    Newcombe paired CI も併記する。
    `key_order_violation` 違反率が SO の両条件いずれかで 1% を超える cell は
    `excluded_by_key_order_gate=True` を立て、`mcnemar_p` / `newcombe_*_pp` を
    NaN にして採点対象外とする (docs/experiment_design.md §8)。
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
    pairs = (
        set(zip(af["runner"], af["model_key"], strict=True))
        & set(zip(rf["runner"], rf["model_key"], strict=True))
    )
    for runner, model_key in sorted(pairs):
        af_g = af[(af["runner"] == runner) & (af["model_key"] == model_key)]
        rf_g = rf[(rf["runner"] == runner) & (rf["model_key"] == model_key)]
        af_r = af_g.set_index("question_id")["correct"]
        rf_r = rf_g.set_index("question_id")["correct"]
        common = af_r.index.intersection(rf_r.index)
        if len(common) == 0:
            continue
        af_c = af_r.loc[common].astype(bool)
        rf_c = rf_r.loc[common].astype(bool)
        e = int(((af_c) & (rf_c)).sum())  # 両方正解
        f = int(((af_c) & (~rf_c)).sum())  # AF のみ正解 (b)
        g = int(((~af_c) & (rf_c)).sum())  # RF のみ正解 (c)
        h = int(((~af_c) & (~rf_c)).sum())  # 両方不正解

        af_viol = _key_order_violation_rate(af_g)
        rf_viol = _key_order_violation_rate(rf_g)
        excluded = (
            af_viol > _KEY_ORDER_GATE_THRESHOLD
            or rf_viol > _KEY_ORDER_GATE_THRESHOLD
        )

        if excluded:
            pval = float("nan")
            ci_lower_pp = float("nan")
            ci_upper_pp = float("nan")
        else:
            table = [[e, f], [g, h]]
            try:
                result = mcnemar(table, exact=True)
                pval = float(result.pvalue)
            except Exception:  # noqa: BLE001
                pval = float("nan")
            # Newcombe paired CI (RF - AF の方向で diff を計算したいので A=RF, B=AF)
            # 引数の意味: e=A正解&B正解, f=A正解&B不正解, g=A不正解&B正解, h=両方不正解
            # A=RF, B=AF にマッピング: e=両方, f=RF only, g=AF only, h=両方不正解
            ci = newcombe_paired_diff_ci(e=e, f=g, g=f, h=h)  # noqa: E501
            ci_lower_pp = ci.lower * 100.0
            ci_upper_pp = ci.upper * 100.0
        af_acc = float(af_c.mean())
        rf_acc = float(rf_c.mean())
        records.append(
            {
                "runner": runner,
                "model_key": model_key,
                "n_paired": int(len(common)),
                "answer_first_acc": af_acc,
                "reasoning_first_acc": rf_acc,
                "diff_pp": (rf_acc - af_acc) * 100.0,
                "newcombe_lower_pp": ci_lower_pp,
                "newcombe_upper_pp": ci_upper_pp,
                "b_af_only": f,
                "c_rf_only": g,
                "mcnemar_p": pval,
                "af_key_order_violation_rate": af_viol,
                "rf_key_order_violation_rate": rf_viol,
                "excluded_by_key_order_gate": excluded,
            }
        )
    return pd.DataFrame.from_records(records).sort_values(["model_key", "runner"])


def runner_spread(df: pd.DataFrame) -> pd.DataFrame:
    """H4: 同条件 × Model で 3 Runner の正解率 max-min を計算。"""
    if df.empty:
        return pd.DataFrame()
    det = df[df["temp_set"] == "det"]
    if det.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for (cond, model_key), g in det.groupby(["condition_id", "model_key"]):
        per_runner = g.groupby("runner")["correct"].mean()
        records.append(
            {
                "condition_id": cond,
                "model_key": model_key,
                **{f"acc_{r}": float(per_runner.get(r, float("nan"))) for r in per_runner.index},
                "max_minus_min_pp": (per_runner.max() - per_runner.min()) * 100.0,
            }
        )
    return pd.DataFrame.from_records(records).sort_values(["model_key", "condition_id"])


def key_order_warnings(df: pd.DataFrame) -> pd.DataFrame:
    """`key_order_violation` フラグ集計。条件 × Model × Runner で件数とレート。"""
    if df.empty:
        return pd.DataFrame()

    structured = df[df["raw_response_keys"].notna()] if "raw_response_keys" in df.columns else df

    if structured.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for (cond, runner, model_key), g in structured.groupby(
        ["condition_id", "runner", "model_key"]
    ):
        total = len(g)
        if "key_order_violation" in g.columns:
            mismatches = int(g["key_order_violation"].fillna(False).astype(bool).sum())
        else:
            mismatches = 0
        records.append(
            {
                "condition_id": cond,
                "runner": runner,
                "model_key": model_key,
                "structured_calls": total,
                "key_order_mismatches": mismatches,
                "violation_rate": (mismatches / total) if total else 0.0,
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["model_key", "condition_id", "runner"]
    )


# ---- summary.md 生成 --------------------------------------------------------


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no data)_\n"
    return (df.to_markdown(index=False, floatfmt=".4f") or "") + "\n"


def _count_errors(errors_path: Path) -> int:
    if not errors_path.exists():
        return 0
    with errors_path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def write_summary(
    run_id: str,
    results_root: Path | str = "results/runs",
) -> Path:
    root = Path(results_root) / run_id
    jsonl = root / "raw.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"raw.jsonl not found: {jsonl}")
    df = load_jsonl(jsonl)
    n_errors = _count_errors(root / "errors.jsonl")
    n_rows = len(df)

    sections: list[str] = []
    sections.append(f"# Summary — `{run_id}`\n")
    meta_path = root / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sections.append(
            f"- stage: `{meta.get('stage')}`\n"
            f"- git_commit: `{meta.get('git_commit')}` (dirty={meta.get('git_dirty')})\n"
            f"- api_version: `{meta.get('api_version')}` / exec_seed: `{meta.get('exec_seed')}`\n"
            f"- conditions.yaml sha256: `{meta.get('conditions_yaml_sha256')}`\n"
        )
        models = meta.get("models") or []
        if models:
            sections.append("\n### Models\n")
            for m in models:
                sections.append(
                    f"- `{m.get('key')}` deployment=`{m.get('deployment')}` "
                    f"reasoning_effort=`{m.get('reasoning_effort')}`\n"
                )

    total_attempts = n_rows + n_errors
    err_rate = (n_errors / total_attempts) if total_attempts else 0.0
    sections.append(
        f"\n### Call counts\n"
        f"- rows in `raw.jsonl`: {n_rows}\n"
        f"- entries in `errors.jsonl`: {n_errors} "
        f"(error rate {err_rate:.4f} of {total_attempts} attempts)\n"
    )

    sections.append("\n## Accuracy by (condition, runner, model_key, temp_set)\n")
    sections.append(_df_to_markdown(accuracy_table(df)))

    sections.append("\n## Self-consistency majority vote accuracy (temp_set=sc)\n")
    sections.append(_df_to_markdown(majority_accuracy_table(df)))

    sections.append(
        "\n## H1 — Order effect (so_reasoning_first vs so_answer_first) [+Newcombe paired CI]\n"
        "_Cell が `excluded_by_key_order_gate=True` の場合、`mcnemar_p` と Newcombe CI は NaN "
        "(SO 両条件いずれかで key_order_violation > 1%; docs/experiment_design.md §8)。_\n"
    )
    sections.append(_df_to_markdown(order_effect_mcnemar(df)))

    sections.append("\n## H4 — Runner spread (max - min, percentage points)\n")
    sections.append(_df_to_markdown(runner_spread(df)))

    sections.append("\n## Key order violations (raw_response_keys vs declared)\n")
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
