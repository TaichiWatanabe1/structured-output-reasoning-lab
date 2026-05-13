# Claude Code Notes for sor-lab

## What this repo is

Structured Output の field 宣言順が LLM 推論精度に与える影響を測る実験基盤。
詳細は [docs/experiment_design.md](docs/experiment_design.md) を最初に読む。

## House rules

- **依存はすべて `==` でピン留め**。`uv add` / `uv lock` で解決した実バージョンを
  pyproject.toml に転記する。`>=` / `~=` / `^=` は使わない。
- **`uv.lock` は必ずコミット**。CI は `uv sync --frozen` を要求する想定。
- **`requires-python = "==3.12.4"`** を変更しない。`.python-version` も同じ。
- **実験条件 (軸1〜4) を変更する場合は必ず `docs/experiment_design.md` を先に更新**
  してからコードを書く (事前登録の原則)。
- **プロンプトは `src/sor_lab/prompts.py` の定数のみを使う**。テストで diff を取り、
  意図しない揺れを検出する。
- **Pydantic スキーマのフィールド宣言順 = 生成順** を実験前提として依存しているため、
  `tests/test_schemas.py` で `model_json_schema()["properties"]` のキー順を assert する。
- **API キーは `.env` に置き `pydantic-settings` でロード**。コード内ハードコード禁止。
- **結果 JSONL は append-only**。同じ `(condition_id, runner, temp_set, try_idx, question_id)` 行は再実行時に skip する。

## 命名規約

- Runner ID: `openai_sdk` / `azure_sdk` / `langchain`
- Condition ID: `plain_direct` / `plain_cot` / `so_answer_first` / `so_reasoning_first` / `so_step_constrained:n_steps=<N>:answer_first=<bool>`
- Temp set: `det` / `sc`
- Run ID: ISO 8601 UTC タイムスタンプ `YYYYMMDDTHHMMSSZ`
