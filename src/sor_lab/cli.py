"""Typer CLI: `sor-lab smoke`, `sor-lab run --stage <name>`, `sor-lab report <run_id>`."""

from __future__ import annotations

from pathlib import Path

import typer

from sor_lab.config import AzureOpenAISettings, load_settings
from sor_lab.experiment import ExperimentConfig, load_config, run_stage


app = typer.Typer(help="Structured Output Reasoning Lab")


def _load_or_exit(
    conditions_yaml: Path,
) -> tuple[ExperimentConfig, AzureOpenAISettings]:
    """`load_config` + `load_settings` を実行し、欠落があれば exit。"""
    try:
        cfg = load_config(conditions_yaml)
    except ValueError as exc:
        typer.echo(f"ERROR: {conditions_yaml} の解析に失敗: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    settings = load_settings()
    if not settings.is_complete():
        missing = ", ".join(settings.missing_fields())
        typer.echo(
            f"ERROR: 環境変数が未設定: {missing}。.env を確認してください。",
            err=True,
        )
        raise typer.Exit(code=2)
    return cfg, settings


@app.command()
def smoke(
    conditions_yaml: Path = typer.Option(
        Path("conditions.yaml"), "--conditions", help="conditions.yaml への path"
    ),
) -> None:
    """1 問 × 全条件で疎通確認 (conditions.yaml の `stages.smoke`)。"""
    cfg, settings = _load_or_exit(conditions_yaml)
    path = run_stage(cfg=cfg, stage_name="smoke", settings=settings)
    typer.echo(f"smoke run wrote: {path}")


@app.command()
def run(
    stage: str = typer.Option(..., "--stage", help="conditions.yaml の stages から選択"),
    conditions_yaml: Path = typer.Option(
        Path("conditions.yaml"), "--conditions", help="conditions.yaml への path"
    ),
    run_id: str | None = typer.Option(
        None, "--run-id", help="既存 run を再開するときに指定"
    ),
) -> None:
    """指定 stage を実行。"""
    cfg, settings = _load_or_exit(conditions_yaml)
    path = run_stage(cfg=cfg, stage_name=stage, settings=settings, run_id=run_id)
    typer.echo(f"stage {stage!r} wrote: {path}")


@app.command()
def report(
    run_id: str = typer.Argument(..., help="results/runs/<run_id> の名前"),
    results_root: Path = typer.Option(Path("results/runs"), "--results-root"),
) -> None:
    """JSONL を集計して summary.md を生成。"""
    from sor_lab.reporting import write_summary

    out = write_summary(run_id=run_id, results_root=results_root)
    typer.echo(f"summary: {out}")


@app.command()
def validate_schemas() -> None:
    """API キーなしで実行できる自己診断 (smoke の事前確認)。

    全 Structured 条件に対し strict JSON Schema を生成して `properties` 順が
    宣言順と一致することを表示。"""
    from sor_lab.runners._schema_utils import schema_property_order, to_strict_schema
    from sor_lab.schemas import field_order, get_schema

    ok = True
    for cond in [
        "so_answer_first",
        "so_reasoning_first",
        "so_step_constrained:n_steps=1:answer_first=False",
        "so_step_constrained:n_steps=3:answer_first=False",
        "so_step_constrained:n_steps=5:answer_first=False",
        "so_step_constrained:n_steps=10:answer_first=False",
        "so_step_constrained:n_steps=5:answer_first=True",
    ]:
        schema_cls = get_schema(cond)
        assert schema_cls is not None
        declared = field_order(schema_cls)
        strict = to_strict_schema(schema_cls)
        actual = schema_property_order(strict)
        match = declared == actual
        ok = ok and match
        typer.echo(
            f"{'OK ' if match else 'NG '}{cond}: declared={declared} strict={actual}"
        )
    if not ok:
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
