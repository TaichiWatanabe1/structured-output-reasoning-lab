"""実験オーケストレーション。conditions.yaml を読んで条件直積をループ実行する。

Append-only JSONL に書き出し、再実行時は既存行を skip して resume できる。
軸5 (Model) を含めた `(condition_id, runner, temp_set, try_idx, model_key, question_id)`
の組ごとに 1 行。実行順は `EXEC_SEED` でシャッフルし、`run_order_idx` を JSONL に記録する。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from sor_lab.config import AzureOpenAISettings, ModelSpec
from sor_lab.datasets import Problem, load_subset
from sor_lab.runners.base import Runner, RunnerResult
from sor_lab.schemas import field_order, get_schema


@dataclass(frozen=True)
class TempSet:
    name: str
    temperature: float
    n_tries: int


@dataclass(frozen=True)
class StageSpec:
    name: str
    conditions: list[str]
    runners: list[str]
    temp_sets: list[str]
    n_questions: int | None  # None = dataset.n


@dataclass(frozen=True)
class ExperimentConfig:
    conditions: list[str]  # 全 condition_id
    runners: list[str]
    temp_sets: dict[str, TempSet]
    model_specs: list[ModelSpec]
    dataset_seed: int
    dataset_n: int
    stages: dict[str, StageSpec]
    raw_yaml: str

    @property
    def yaml_sha256(self) -> str:
        return hashlib.sha256(self.raw_yaml.encode("utf-8")).hexdigest()


def _parse_model_specs(raw_models: Any) -> list[ModelSpec]:
    """`conditions.yaml` の `models:` セクションを `ModelSpec` のリストに変換。"""
    if not raw_models:
        raise ValueError(
            "conditions.yaml must declare at least one model under 'models:'"
        )
    specs: list[ModelSpec] = []
    seen_keys: set[str] = set()
    for entry in raw_models:
        key = entry.get("key")
        deployment = entry.get("deployment")
        reasoning_effort = entry.get("reasoning_effort")
        if not key or not isinstance(key, str):
            raise ValueError(f"models entry missing 'key': {entry!r}")
        if not deployment or not isinstance(deployment, str):
            raise ValueError(
                f"models entry {key!r} has empty or invalid 'deployment'"
            )
        if key in seen_keys:
            raise ValueError(f"duplicate model key in conditions.yaml: {key!r}")
        seen_keys.add(key)
        re_val: str | None
        if reasoning_effort is None or reasoning_effort == "":
            re_val = None
        elif isinstance(reasoning_effort, str):
            re_val = reasoning_effort
        else:
            raise ValueError(
                f"models entry {key!r} has invalid reasoning_effort: "
                f"{reasoning_effort!r}"
            )
        specs.append(ModelSpec(key=key, deployment=deployment, reasoning_effort=re_val))
    return specs


def load_config(path: str | os.PathLike[str] = "conditions.yaml") -> ExperimentConfig:
    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw)

    conditions = [c["id"] for c in data["conditions"]]
    runners = list(data["runners"])
    temp_sets = {
        name: TempSet(name=name, temperature=v["temperature"], n_tries=v["n_tries"])
        for name, v in data["temp_sets"].items()
    }
    model_specs = _parse_model_specs(data.get("models"))
    dataset = data["dataset"]
    stages_raw = data.get("stages", {})
    stages: dict[str, StageSpec] = {}
    for sname, sdef in stages_raw.items():
        stages[sname] = StageSpec(
            name=sname,
            conditions=list(sdef.get("conditions", conditions)),
            runners=list(sdef.get("runners", runners)),
            temp_sets=list(sdef.get("temp_sets", list(temp_sets.keys()))),
            n_questions=sdef.get("n_questions"),
        )

    return ExperimentConfig(
        conditions=conditions,
        runners=runners,
        temp_sets=temp_sets,
        model_specs=model_specs,
        dataset_seed=int(dataset["seed"]),
        dataset_n=int(dataset["n"]),
        stages=stages,
        raw_yaml=raw,
    )


def build_runner(
    name: str, settings: AzureOpenAISettings, model_spec: ModelSpec
) -> Runner:
    """Runner ID + model_spec から具体実装を生成する。"""
    if name == "openai_sdk":
        from sor_lab.runners.openai_sdk import OpenAISdkRunner

        return OpenAISdkRunner(settings, model_spec)
    if name == "azure_sdk":
        from sor_lab.runners.azure_sdk import AzureSdkRunner

        return AzureSdkRunner(settings, model_spec)
    if name == "langchain":
        from sor_lab.runners.langchain import LangChainRunner

        return LangChainRunner(settings, model_spec)
    raise KeyError(f"unknown runner: {name!r}")


# ---- run loop ----------------------------------------------------------------


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_info() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        return sha, bool(status.strip())
    except Exception:
        return "unknown", False


def write_meta(
    run_dir: Path,
    *,
    cfg: ExperimentConfig,
    stage: StageSpec,
    settings: AzureOpenAISettings,
    model_specs: list[ModelSpec],
    extra: dict[str, Any] | None = None,
) -> None:
    sha, dirty = _git_info()
    meta = {
        "run_id": run_dir.name,
        "stage": stage.name,
        "git_commit": sha,
        "git_dirty": dirty,
        "python_version": sys.version,
        "api_version": settings.api_version,
        "exec_seed": settings.exec_seed,
        "models": [
            {
                "key": m.key,
                "deployment": m.deployment,
                "reasoning_effort": m.reasoning_effort,
            }
            for m in model_specs
        ],
        "conditions_yaml_sha256": cfg.yaml_sha256,
        "stage_spec": asdict(stage),
        "command": " ".join(sys.argv),
    }
    if extra:
        meta.update(extra)
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "conditions.yaml").write_text(cfg.raw_yaml, encoding="utf-8")


# ---- row dedup / resume ------------------------------------------------------

# skip キーは (condition_id, runner, temp_set, try_idx, model_key, question_id)
_RowKey = tuple[str, str, str, int, str, str]


def _row_key(row: dict[str, Any]) -> _RowKey:
    return (
        row["condition_id"],
        row["runner"],
        row["temp_set"],
        int(row["try_idx"]),
        row.get("model_key", ""),
        row["question_id"],
    )


def _load_existing_keys(jsonl_path: Path) -> set[_RowKey]:
    if not jsonl_path.exists():
        return set()
    keys: set[_RowKey] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add(_row_key(row))
    return keys


# ---- key order quality gate --------------------------------------------------


def _declared_field_order(condition_id: str) -> list[str] | None:
    """Structured 条件の declared field order を返す。Plain 条件は None。"""
    schema_cls = get_schema(condition_id)
    if schema_cls is None:
        return None
    return field_order(schema_cls)


def _key_order_violation(
    condition_id: str, raw_response_keys: list[str] | None
) -> bool:
    """Structured 出力の property 順が declared field order と一致しなければ True。"""
    declared = _declared_field_order(condition_id)
    if declared is None:
        return False  # Plain 条件は対象外
    if raw_response_keys is None:
        return False  # parse 失敗時は別途 parse_failed で扱う
    return list(raw_response_keys) != declared


def _serialize_result(
    *,
    run_id: str,
    condition_id: str,
    runner_name: str,
    model_spec: ModelSpec,
    temp: TempSet,
    try_idx: int,
    problem: Problem,
    system_prompt: str,
    user_prompt: str,
    result: RunnerResult,
    correct: bool,
    run_order_idx: int,
) -> dict[str, Any]:
    from sor_lab.evaluation import count_reasoning_steps

    return {
        "run_id": run_id,
        "condition_id": condition_id,
        "runner": runner_name,
        "model_key": model_spec.key,
        "deployment": model_spec.deployment,
        "reasoning_effort": model_spec.reasoning_effort,
        "model_string": result.model_string,
        "temp_set": temp.name,
        "temperature": temp.temperature,
        "try_idx": try_idx,
        "question_id": problem.question_id,
        "question": problem.question,
        "gold_answer": problem.gold_answer,
        "prompt": {"system": system_prompt, "user": user_prompt},
        "raw_response": result.raw_response,
        "raw_response_keys": result.raw_response_keys,
        "parsed_answer": result.parsed_answer,
        "parsed_reasoning": result.parsed_reasoning,
        "parsed_reasoning_step_count": count_reasoning_steps(result.parsed_reasoning),
        "parse_failed": result.parse_failed,
        "correct": correct,
        "key_order_violation": _key_order_violation(condition_id, result.raw_response_keys),
        "latency_ms": result.latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "run_order_idx": run_order_idx,
    }


# ---- plan construction -------------------------------------------------------


@dataclass(frozen=True)
class CallSpec:
    """シャッフル対象になる 1 コールの仕様。"""

    condition_id: str
    runner_name: str
    temp_set_name: str
    try_idx: int
    model_key: str
    question_id: str


def build_call_list(
    cfg: ExperimentConfig,
    stage: StageSpec,
    model_specs: list[ModelSpec],
    problems: list[Problem],
) -> list[CallSpec]:
    """ステージが含む全コールをフラットなリストにする (シャッフル前)。"""
    plan: list[CallSpec] = []
    for cond in stage.conditions:
        for runner_name in stage.runners:
            for ts_name in stage.temp_sets:
                temp = cfg.temp_sets[ts_name]
                for try_idx in range(temp.n_tries):
                    for model_spec in model_specs:
                        for problem in problems:
                            plan.append(
                                CallSpec(
                                    condition_id=cond,
                                    runner_name=runner_name,
                                    temp_set_name=ts_name,
                                    try_idx=try_idx,
                                    model_key=model_spec.key,
                                    question_id=problem.question_id,
                                )
                            )
    return plan


def shuffle_call_list(plan: list[CallSpec], seed: int) -> list[CallSpec]:
    """`random.Random(seed).shuffle` で実行順を確定する。"""
    rng = random.Random(seed)
    shuffled = list(plan)
    rng.shuffle(shuffled)
    return shuffled


def run_stage(
    *,
    cfg: ExperimentConfig,
    stage_name: str,
    settings: AzureOpenAISettings,
    run_id: str | None = None,
    results_root: str | os.PathLike[str] = "results/runs",
) -> Path:
    """指定 stage を実行し、`results/runs/<run_id>/raw.jsonl` を作成して path を返す。"""
    from sor_lab.prompts import get_prompt

    if stage_name not in cfg.stages:
        raise KeyError(f"unknown stage: {stage_name!r}")
    stage = cfg.stages[stage_name]

    n = stage.n_questions if stage.n_questions is not None else cfg.dataset_n
    problems = load_subset(n=n, seed=cfg.dataset_seed)
    problems_by_id: dict[str, Problem] = {p.question_id: p for p in problems}

    model_specs = cfg.model_specs
    if not model_specs:
        raise RuntimeError(
            "No models configured. Add at least one entry to the 'models:' "
            "section of conditions.yaml."
        )
    model_specs_by_key: dict[str, ModelSpec] = {m.key: m for m in model_specs}

    rid = run_id or new_run_id()
    run_dir = Path(results_root) / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "raw.jsonl"
    errors_path = run_dir / "errors.jsonl"

    write_meta(
        run_dir,
        cfg=cfg,
        stage=stage,
        settings=settings,
        model_specs=model_specs,
    )

    existing = _load_existing_keys(jsonl_path)

    # コール列の構築とシャッフル
    plan = build_call_list(cfg, stage, model_specs, problems)
    plan = shuffle_call_list(plan, seed=settings.exec_seed)

    # Runner cache key: (runner_name, model_key)
    runners_cache: dict[tuple[str, str], Runner] = {}

    started = time.perf_counter()
    n_done = 0
    n_skipped = 0
    n_errors = 0
    with (
        jsonl_path.open("a", encoding="utf-8") as out,
        errors_path.open("a", encoding="utf-8") as errors_out,
    ):
        for run_order_idx, call in enumerate(plan):
            key: _RowKey = (
                call.condition_id,
                call.runner_name,
                call.temp_set_name,
                call.try_idx,
                call.model_key,
                call.question_id,
            )
            if key in existing:
                n_skipped += 1
                continue

            model_spec = model_specs_by_key[call.model_key]
            cache_key = (call.runner_name, call.model_key)
            runner = runners_cache.get(cache_key)
            if runner is None:
                runner = build_runner(call.runner_name, settings, model_spec)
                runners_cache[cache_key] = runner

            temp = cfg.temp_sets[call.temp_set_name]
            problem = problems_by_id[call.question_id]
            prompt = get_prompt(call.condition_id)
            system_prompt, user_prompt = prompt.render(problem.question)

            # SC 試行は seed 未指定で独立サンプルを取る。det (n_tries=1) のみ
            # `dataset_seed` を渡して再現性を確保する。docs/experiment_design.md §5 参照。
            call_seed = cfg.dataset_seed if temp.n_tries == 1 else None
            try:
                result = runner.run(
                    condition_id=call.condition_id,
                    question=problem.question,
                    temperature=temp.temperature,
                    seed=call_seed,
                )
            except Exception as exc:  # noqa: BLE001
                err_row = {
                    "run_id": rid,
                    "condition_id": call.condition_id,
                    "runner": call.runner_name,
                    "model_key": call.model_key,
                    "temp_set": call.temp_set_name,
                    "try_idx": call.try_idx,
                    "question_id": call.question_id,
                    "run_order_idx": run_order_idx,
                    "error": f"{type(exc).__name__}: {exc}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                errors_out.write(json.dumps(err_row, ensure_ascii=False) + "\n")
                errors_out.flush()
                print(f"[error] {err_row['error']}", file=sys.stderr)
                n_errors += 1
                continue

            correct = (
                result.parsed_answer is not None
                and result.parsed_answer == problem.gold_answer
            )
            row = _serialize_result(
                run_id=rid,
                condition_id=call.condition_id,
                runner_name=call.runner_name,
                model_spec=model_spec,
                temp=temp,
                try_idx=call.try_idx,
                problem=problem,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                result=result,
                correct=correct,
                run_order_idx=run_order_idx,
            )
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            n_done += 1

    elapsed = time.perf_counter() - started
    print(
        f"[run_stage] stage={stage_name} run_id={rid} done={n_done} "
        f"skipped={n_skipped} errors={n_errors} total_planned={len(plan)} "
        f"elapsed={elapsed:.1f}s",
        file=sys.stderr,
    )
    return jsonl_path


__all__ = [
    "TempSet",
    "StageSpec",
    "ExperimentConfig",
    "CallSpec",
    "load_config",
    "build_runner",
    "build_call_list",
    "shuffle_call_list",
    "new_run_id",
    "run_stage",
]
