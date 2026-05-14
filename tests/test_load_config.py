"""`load_config` の `models:` セクション parse を検証。"""

from __future__ import annotations

from pathlib import Path

import pytest

from sor_lab.experiment import load_config


_BASE_YAML = """
conditions:
  - id: plain_direct
runners: [openai_sdk]
temp_sets:
  det:
    temperature: 0.0
    n_tries: 1
dataset:
  name: mgsm
  config: ja
  split: test
  seed: 42
  n: 1
"""


def _write(tmp_path: Path, extra: str) -> Path:
    p = tmp_path / "conditions.yaml"
    p.write_text(_BASE_YAML + extra, encoding="utf-8")
    return p


def test_load_config_parses_models_section(tmp_path: Path) -> None:
    cfg = load_config(
        _write(
            tmp_path,
            """
models:
  - key: gpt-4.1
    deployment: dep-41
    reasoning_effort: null
  - key: gpt-5
    deployment: dep-5
    reasoning_effort: minimal
""",
        )
    )
    assert [m.key for m in cfg.model_specs] == ["gpt-4.1", "gpt-5"]
    assert cfg.model_specs[0].deployment == "dep-41"
    assert cfg.model_specs[0].reasoning_effort is None
    assert cfg.model_specs[1].reasoning_effort == "minimal"


def test_load_config_missing_models_section_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="models:"):
        load_config(_write(tmp_path, ""))


def test_load_config_empty_models_list_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="models:"):
        load_config(_write(tmp_path, "\nmodels: []\n"))


def test_load_config_empty_deployment_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="deployment"):
        load_config(
            _write(
                tmp_path,
                """
models:
  - key: gpt-4.1
    deployment: ""
    reasoning_effort: null
""",
            )
        )


def test_load_config_duplicate_key_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        load_config(
            _write(
                tmp_path,
                """
models:
  - key: gpt-4.1
    deployment: dep-a
    reasoning_effort: null
  - key: gpt-4.1
    deployment: dep-b
    reasoning_effort: null
""",
            )
        )


def test_load_config_reasoning_effort_omitted_is_none(tmp_path: Path) -> None:
    """reasoning_effort: キーを書かなくても None になる。"""
    cfg = load_config(
        _write(
            tmp_path,
            """
models:
  - key: gpt-4.1
    deployment: dep-41
""",
        )
    )
    assert cfg.model_specs[0].reasoning_effort is None
