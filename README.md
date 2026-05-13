# Structured Output Reasoning Lab

LLM の **Structured Output におけるフィールド宣言順序が推論精度に与える影響** を、
再現可能な形で検証する実験基盤。

仮説: Pydantic / JSON Schema のフィールド順序が生成トークン順を決定する
(自己回帰生成) ため、`reasoning` を `answer` より先に置くと Chain-of-Thought 効果が
得られ、逆だと答えを先に確定してしまうので CoT 効果がゼロになる。

## 検証する軸

1. **Output Format × Prompt** (5 条件): plain_direct / plain_cot / so_answer_first / so_reasoning_first / so_step_constrained
2. **Step 数** (StepConstrained のみ、{1, 3, 5, 10})
3. **Runner** (3 種): `openai_sdk` (`openai` 公式), `azure_sdk` (`azure-ai-inference`), `langchain` (`langchain-openai`)
4. **Temperature × N**: `det` (T=0.0, n=1) と `sc` (T=0.7, n=5)

データセット: **GSM8K** test split から seed=42 で 50 問固定サブセット。
モデル: **Azure OpenAI** デプロイメント。

詳細は [docs/experiment_design.md](docs/experiment_design.md) を参照。

## セットアップ

```bash
# Python 3.12.4 が必要 (.python-version で固定)
uv sync --frozen

# .env を作成
cp .env.example .env
# AZURE_OPENAI_API_KEY / ENDPOINT / API_VERSION / DEPLOYMENT_NAME を埋める
```

## 実行

```bash
# テスト
uv run pytest

# 疎通確認 (1 問 × 全条件)
uv run sor-lab smoke

# Stage B-1: 順序 + Plain ベースライン
uv run sor-lab run --stage b1

# Stage B-2: ステップ数効果
uv run sor-lab run --stage b2

# Stage B-3: 安定性 (temperature=0.7×5)
uv run sor-lab run --stage b3

# レポート生成
uv run sor-lab report <run_id>
```

## ディレクトリ

```
docs/                       # 実験デザイン文書 (再現性の核)
src/sor_lab/                # 実装
  schemas.py                # Pydantic スキーマ (AnswerFirst / ReasoningFirst / StepConstrained)
  prompts.py                # 5 条件のプロンプト定数
  datasets.py               # GSM8K loader
  evaluation.py             # 数値抽出 + 正解判定
  runners/                  # 3 種の SDK Runner
  experiment.py             # 実験オーケストレーション
  reporting.py              # JSONL -> summary.md
  cli.py                    # Typer CLI
tests/                      # pytest
conditions.yaml             # 実験条件の宣言ファイル
results/runs/<run_id>/      # 各実行の生データ
```

## 再現性

- 依存は `pyproject.toml` で `==` ピン、`uv.lock` をコミット
- `requires-python==3.12.4` で Python パッチバージョンまで固定
- 各 run の `meta.json` に git_commit / conditions.yaml SHA / API version / 解決済みモデル名 を記録
- 結果 JSONL に `raw_response` と `raw_response_keys` を保存し、宣言順とキー順の一致を後検証可能
