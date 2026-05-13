.PHONY: help sync lint typecheck test fmt ci validate-schemas smoke run run-b1 run-b2 run-b3 report

STAGE ?=
RUN_ID ?=

help:
	@echo "Dev targets:"
	@echo "  sync              - uv sync --frozen --all-groups"
	@echo "  lint              - ruff check"
	@echo "  typecheck         - mypy (strict)"
	@echo "  test              - pytest"
	@echo "  fmt               - ruff format + ruff check --fix"
	@echo "  ci                - lint + typecheck + test (CI と同じ順序)"
	@echo ""
	@echo "Experiment targets:"
	@echo "  validate-schemas  - API キー不要の自己診断 (smoke の事前確認)"
	@echo "  smoke             - 1 問 × 全条件で疎通確認"
	@echo "  run STAGE=<name>  - 任意 stage を実行 (例: make run STAGE=b1)"
	@echo "  run-b1            - stage b1 (軸1: 5 条件 × det)"
	@echo "  run-b2            - stage b2 (軸2: StepConstrained × det)"
	@echo "  run-b3            - stage b3 (全条件 × sc: 主実験, 重い)"
	@echo "  report RUN_ID=<id>- results/runs/<id> を集計して summary.md 生成"

sync:
	uv sync --frozen --all-groups

lint:
	uv run ruff check src/ tests/

typecheck:
	uv run mypy src/

test:
	uv run pytest -q

fmt:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

ci: lint typecheck test

validate-schemas:
	uv run sor-lab validate-schemas

smoke:
	uv run sor-lab smoke

run:
	@if [ -z "$(STAGE)" ]; then echo "ERROR: STAGE is required. e.g. make run STAGE=b1"; exit 2; fi
	uv run sor-lab run --stage $(STAGE)

run-b1:
	uv run sor-lab run --stage b1

run-b2:
	uv run sor-lab run --stage b2

run-b3:
	uv run sor-lab run --stage b3

report:
	@if [ -z "$(RUN_ID)" ]; then echo "ERROR: RUN_ID is required. e.g. make report RUN_ID=20260513T012345Z"; exit 2; fi
	uv run sor-lab report $(RUN_ID)
