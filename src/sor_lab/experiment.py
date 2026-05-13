"""実験オーケストレーション。conditions.yaml を読んで条件直積をループ実行する。

Append-only JSONL に書き出し、再実行時は既存行を skip して resume できる。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from sor_lab.config import AzureOpenAISettings
from sor_lab.datasets import Problem, load_subset
from sor_lab.runners.base import Runner, RunnerResult


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
    dataset_seed: int
    dataset_n: int
    stages: dict[str, StageSpec]
    raw_yaml: str

    @property
    def yaml_sha256(self) -> str:
        return hashlib.sha256(self.raw_yaml.encode("utf-8")).hexdigest()


def load_config(path: str | os.PathLike[str] = "conditions.yaml") -> ExperimentConfig:
    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw)

    conditions = [c["id"] for c in data["conditions"]]
    runners = list(data["runners"])
    temp_sets = {
        name: TempSet(name=name, temperature=v["temperature"], n_tries=v["n_tries"])
        for name, v in data["temp_sets"].items()
    }
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
        dataset_seed=int(dataset["seed"]),
        dataset_n=int(dataset["n"]),
        stages=stages,
        raw_yaml=raw,
    )


def build_runner(name: str, settings: AzureOpenAISettings) -> Runner:
    """Runner ID から具体実装を生成する。"""
    if name == "openai_sdk":
        from sor_lab.runners.openai_sdk import OpenAISdkRunner

        return OpenAISdkRunner(settings)
    if name == "azure_sdk":
        from sor_lab.runners.azure_sdk import AzureSdkRunner

        return AzureSdkRunner(settings)
    if name == "langchain":
        from sor_lab.runners.langchain import LangChainRunner

        return LangChainRunner(settings)
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
    deployment: str,
    api_version: str,
    extra: dict[str, Any] | None = None,
) -> None:
    sha, dirty = _git_info()
    meta = {
        "run_id": run_dir.name,
        "stage": stage.name,
        "git_commit": sha,
        "git_dirty": dirty,
        "python_version": sys.version,
        "deployment": deployment,
        "api_version": api_version,
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


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, int, str]:
    return (
        row["condition_id"],
        row["runner"],
        row["temp_set"],
        int(row["try_idx"]),
        row["question_id"],
    )


def _load_existing_keys(jsonl_path: Path) -> set[tuple[str, str, str, int, str]]:
    if not jsonl_path.exists():
        return set()
    keys: set[tuple[str, str, str, int, str]] = set()
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


def _serialize_result(
    *,
    run_id: str,
    condition_id: str,
    runner_name: str,
    temp: TempSet,
    try_idx: int,
    problem: Problem,
    system_prompt: str,
    user_prompt: str,
    result: RunnerResult,
    correct: bool,
) -> dict[str, Any]:
    from sor_lab.evaluation import count_reasoning_steps

    return {
        "run_id": run_id,
        "condition_id": condition_id,
        "runner": runner_name,
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
        "latency_ms": result.latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }


def iter_plan(
    cfg: ExperimentConfig,
    stage: StageSpec,
    problems: list[Problem],
) -> Iterable[tuple[str, str, TempSet, int, Problem]]:
    """`(condition_id, runner_name, temp, try_idx, problem)` を yield。"""
    for cond in stage.conditions:
        for runner_name in stage.runners:
            for ts_name in stage.temp_sets:
                temp = cfg.temp_sets[ts_name]
                for try_idx in range(temp.n_tries):
                    for problem in problems:
                        yield cond, runner_name, temp, try_idx, problem


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

    rid = run_id or new_run_id()
    run_dir = Path(results_root) / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "raw.jsonl"

    write_meta(
        run_dir,
        cfg=cfg,
        stage=stage,
        deployment=settings.deployment_name,
        api_version=settings.api_version,
    )

    existing = _load_existing_keys(jsonl_path)

    # Runner はステージ全体で同じ設定なので 1 度だけ作る
    runners_cache: dict[str, Runner] = {}

    started = time.perf_counter()
    n_done = 0
    n_skipped = 0
    with jsonl_path.open("a", encoding="utf-8") as out:
        for cond, runner_name, temp, try_idx, problem in iter_plan(cfg, stage, problems):
            key = (cond, runner_name, temp.name, try_idx, problem.question_id)
            if key in existing:
                n_skipped += 1
                continue

            runner = runners_cache.get(runner_name)
            if runner is None:
                runner = build_runner(runner_name, settings)
                runners_cache[runner_name] = runner

            prompt = get_prompt(cond)
            system_prompt, user_prompt = prompt.render(problem.question)

            try:
                result = runner.run(
                    condition_id=cond,
                    question=problem.question,
                    temperature=temp.temperature,
                    seed=cfg.dataset_seed,
                )
            except Exception as exc:  # noqa: BLE001
                # 失敗は記録のみ、次に進む (再実行時に再試行される)
                err_row = {
                    "run_id": rid,
                    "condition_id": cond,
                    "runner": runner_name,
                    "temp_set": temp.name,
                    "temperature": temp.temperature,
                    "try_idx": try_idx,
                    "question_id": problem.question_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"[error] {err_row['error']}", file=sys.stderr)
                continue

            correct = (
                result.parsed_answer is not None
                and result.parsed_answer == problem.gold_answer
            )
            row = _serialize_result(
                run_id=rid,
                condition_id=cond,
                runner_name=runner_name,
                temp=temp,
                try_idx=try_idx,
                problem=problem,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                result=result,
                correct=correct,
            )
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            n_done += 1

    elapsed = time.perf_counter() - started
    print(
        f"[run_stage] stage={stage_name} run_id={rid} done={n_done} "
        f"skipped={n_skipped} elapsed={elapsed:.1f}s",
        file=sys.stderr,
    )
    return jsonl_path


__all__ = [
    "TempSet",
    "StageSpec",
    "ExperimentConfig",
    "load_config",
    "build_runner",
    "new_run_id",
    "run_stage",
    "iter_plan",
]
