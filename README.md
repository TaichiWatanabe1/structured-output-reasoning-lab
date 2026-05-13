# Structured Output Reasoning Lab

LLM の **Structured Output におけるフィールド宣言順序が推論精度に与える影響** を、
再現可能な形で検証する実験基盤。

仮説: Pydantic / JSON Schema のフィールド順序が生成トークン順を決定する
(自己回帰生成) ため、`reasoning` を `answer` より先に置くと Chain-of-Thought 効果が
得られ、逆だと答えを先に確定してしまうので CoT 効果がゼロになる。

## 検証する軸

1. **Output Format × Prompt** (5 条件): plain_direct / plain_cot / so_answer_first / so_reasoning_first / so_step_constrained
2. **Step 数** (StepConstrained のみ、`n_steps ∈ {1, 3, 5, 10}` × `answer_first=False` 固定)
3. **Runner** (3 種): `openai_sdk` (`openai` 公式), `azure_sdk` (`azure-ai-inference`), `langchain` (`langchain-openai`)
4. **Temperature × N**: `det` (T=0.0, n=1) と `sc` (T=0.7, n=5)
5. **Model** (3 種): `gpt-4.1`, `gpt-4.1-mini`, `gpt-5` (`reasoning_effort=minimal` 固定)

データセット: **MGSM-ja** 全 250 問 (GSM8K の人手日本語翻訳版)。言語は日本語固定 (本番が JP 向けのため)。
モデル: **Azure OpenAI** に上記 3 デプロイメント。

副次的な META 仮説として、`gpt-5` で順序効果が縮小すれば「reasoning architecture が
SO field 順序効果を抑制する」を検証する。

詳細は [docs/experiment_design.md](docs/experiment_design.md) を参照。

## セットアップ

```bash
# Python 3.12.4 が必要 (.python-version で固定)
uv sync --frozen

# .env を作成
cp .env.example .env
# AZURE_OPENAI_API_KEY / ENDPOINT / API_VERSION
# AZURE_OPENAI_DEPLOYMENT_NAME_GPT41 / _GPT41_MINI / _GPT5
# AZURE_OPENAI_REASONING_EFFORT=minimal  (gpt-5 用)
# EXEC_SEED=42                            (省略可)
```

## 実行

```bash
# テスト
uv run pytest

# 疎通確認 (1 問 × 主要4条件 × 3 model = 12 calls)
uv run sor-lab smoke

# b1: Plain + SO 基本 (4 cond) × det × 3 model = 9,000 calls
uv run sor-lab run --stage b1

# b2: StepConstrained (4 sub-cond) × det × 3 model = 9,000 calls
uv run sor-lab run --stage b2

# b3: 全 8 cond × sc (T=0.7×5) × 3 model = 90,000 calls
uv run sor-lab run --stage b3

# レポート生成
uv run sor-lab report <run_id>
```

## ディレクトリ

```
docs/                       # 実験デザイン文書 (再現性の核)
src/sor_lab/                # 実装
  schemas.py                # Pydantic スキーマ (AnswerFirst / ReasoningFirst / StepConstrained)
  prompts.py                # 5 条件の日本語プロンプト定数
  datasets.py               # MGSM-ja loader
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
- 各 run の `meta.json` に git_commit / conditions.yaml SHA / API version / 3 model のデプロイ名 & `model_string` / `reasoning_effort` / `EXEC_SEED` を記録
- 結果 JSONL に `raw_response` と `raw_response_keys` を保存し、宣言順とキー順の一致を後検証可能 (runtime quality gate `key_order_violation` + 1% 閾値で H1 主検定の除外を自動化)
- 実行順は `random.Random(EXEC_SEED).shuffle(...)` で stage 内シャッフル、`run_order_idx` を JSONL に記録
