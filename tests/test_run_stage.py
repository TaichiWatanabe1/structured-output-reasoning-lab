"""`run_stage` の SC seed 分岐と `errors.jsonl` 永続化を fake runner で検証。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sor_lab import experiment
from sor_lab.config import AzureOpenAISettings, ModelSpec
from sor_lab.datasets import Problem
from sor_lab.experiment import ExperimentConfig, StageSpec, TempSet, run_stage
from sor_lab.runners.base import RunnerResult


class _FakeRunner:
    """`Runner` Protocol 互換の fake。受け取った `seed` を記録する。"""

    name = "fake"

    def __init__(self, *, raise_on: tuple[int, ...] = ()) -> None:
        self.calls: list[dict[str, Any]] = []
        self._raise_on = set(raise_on)
        self._n = 0

    def run(
        self,
        condition_id: str,
        question: str,
        temperature: float,
        seed: int | None = None,
    ) -> RunnerResult:
        self.calls.append(
            {
                "condition_id": condition_id,
                "question": question,
                "temperature": temperature,
                "seed": seed,
            }
        )
        idx = self._n
        self._n += 1
        if idx in self._raise_on:
            raise RuntimeError(f"injected failure #{idx}")
        return RunnerResult(
            raw_response=json.dumps({"reasoning": "x", "answer": 1}),
            raw_response_keys=["reasoning", "answer"],
            parsed_answer=1,
            parsed_reasoning="x",
            parse_failed=False,
            latency_ms=1,
            prompt_tokens=1,
            completion_tokens=1,
            model_string="fake-model",
        )


def _settings() -> AzureOpenAISettings:
    return AzureOpenAISettings(
        AZURE_OPENAI_API_KEY="k",
        AZURE_OPENAI_ENDPOINT="https://x",
        EXEC_SEED=0,
    )


def _cfg() -> ExperimentConfig:
    return ExperimentConfig(
        conditions=["so_reasoning_first"],
        runners=["fake"],
        temp_sets={
            "det": TempSet("det", 0.0, 1),
            "sc": TempSet("sc", 0.7, 3),
        },
        model_specs=[ModelSpec("gpt-4.1", "dep-41", None)],
        dataset_seed=42,
        dataset_n=1,
        stages={
            "test": StageSpec(
                name="test",
                conditions=["so_reasoning_first"],
                runners=["fake"],
                temp_sets=["det", "sc"],
                n_questions=1,
            )
        },
        raw_yaml="",
    )


def _problems() -> list[Problem]:
    return [
        Problem(question_id="mgsm_ja_0", question="1+1?", gold_answer=2, reference="")
    ]


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> _FakeRunner:
    fake = _FakeRunner()
    monkeypatch.setattr(experiment, "build_runner", lambda *a, **k: fake)
    monkeypatch.setattr(experiment, "load_subset", lambda n, seed: _problems())
    return fake


def test_run_stage_passes_seed_only_for_det(
    patched: _FakeRunner, tmp_path: Path
) -> None:
    run_stage(
        cfg=_cfg(),
        stage_name="test",
        settings=_settings(),
        run_id="r1",
        results_root=tmp_path,
    )
    det_seeds = [c["seed"] for c in patched.calls if c["temperature"] == 0.0]
    sc_seeds = [c["seed"] for c in patched.calls if c["temperature"] == 0.7]
    assert det_seeds and all(s == 42 for s in det_seeds), det_seeds
    assert sc_seeds and all(s is None for s in sc_seeds), sc_seeds


def test_run_stage_persists_errors_to_errors_jsonl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FakeRunner(raise_on=(0, 1))  # 最初の 2 コールだけ失敗
    monkeypatch.setattr(experiment, "build_runner", lambda *a, **k: fake)
    monkeypatch.setattr(experiment, "load_subset", lambda n, seed: _problems())

    run_stage(
        cfg=_cfg(),
        stage_name="test",
        settings=_settings(),
        run_id="r1",
        results_root=tmp_path,
    )
    raw = tmp_path / "r1" / "raw.jsonl"
    errors = tmp_path / "r1" / "errors.jsonl"
    assert errors.exists()
    error_lines = [
        json.loads(line) for line in errors.read_text().splitlines() if line.strip()
    ]
    assert len(error_lines) == 2
    assert "injected failure" in error_lines[0]["error"]
    assert "timestamp" in error_lines[0]
    # raw.jsonl には失敗行は書かれない (成功した残り 2 コール分)
    raw_lines = [
        json.loads(line) for line in raw.read_text().splitlines() if line.strip()
    ]
    assert len(raw_lines) == 4 - 2
