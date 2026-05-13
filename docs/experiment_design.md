# Experiment Design — Structured Output Field Order and Reasoning Accuracy

> 本書は **データ収集前に確定**させる事前登録 (pre-registration) 文書です。
> コード変更が条件・分析手法に影響を与える場合は、本書を先に更新してください。

## 1. Abstract

大規模言語モデル (LLM) は左から右へ自己回帰的にトークンを生成するため、生成順序が
そのまま条件付け確率の連鎖を決定する (Chain-of-Thought; Wei et al., 2022)。
Pydantic / JSON Schema による Structured Output を用いる場合、フィールドの宣言順序が
そのまま生成順序となるため、`answer` を先に置くか `reasoning` を先に置くかで
最終解答の精度が大きく変わると予想される。

本研究では Azure OpenAI 上の単一デプロイメントに対し、(i) Structured Output の
フィールド順序、(ii) reasoning ステップ数、(iii) クライアントライブラリ
(OpenAI 公式 SDK / Microsoft Azure AI Inference SDK / LangChain)、(iv) 温度設定の
4 軸で GSM8K サブセット 50 問の解答精度を測定し、上記仮説を検証する。

## 2. Background

- **Self-Attention の左から右の依存**: Transformer Decoder の自己回帰生成では、
  時点 t のトークン分布は時点 t 未満の全トークンに条件付けられる。
- **Chain-of-Thought (Wei et al., 2022)**: 推論過程を中間トークンとして明示すると
  最終解答精度が向上する。これは中間トークンが後続の Attention に組み込まれることで、
  実効的な計算深度が増えるためと解釈されている。
- **Show Your Work (Nye et al., 2021)**: 算術スクラッチパッドを書かせる手法。
- **Self-Consistency (Wang et al., 2022)**: 温度を上げて複数サンプルを多数決すると
  精度が向上する。
- **Structured Output の文法強制**: OpenAI の `response_format` に `strict: true` を
  渡すと、JSON Schema が文法レベルで強制され、`properties` の宣言順がそのまま
  生成順序になる (本研究で smoke test により確認)。

## 3. Research Questions / Hypotheses

| ID | 仮説 | 評価指標 |
|----|------|----------|
| H1 | `ReasoningFirst` の正解率は `AnswerFirst` を有意に上回る (想定 +10pt 以上) | McNemar 検定の p 値、正解率差 |
| H2 | `so_reasoning_first` の正解率は古典 `plain_cot` と概ね同等 (±3pt)。`so_answer_first` は `plain_direct` と同等 | 正解率差と 95% CI の重なり |
| H3 | `StepConstrained` の正解率は `n_steps ∈ {1, 3, 5, 10}` に対し単調増加または飽和 | Spearman 順位相関 |
| H4 | `openai_sdk` / `azure_sdk` / `langchain` の 3 Runner 間で同条件の正解率差は ±2pt 以内 | 3 Runner の最大-最小差 |
| H5 | `temperature=0.7 × 5 試行` の多数決正解率は `temperature=0` を上回る (CoT 系で顕著) | 多数決 vs 単発正解率差 |

## 4. Materials

- **Dataset**: [GSM8K](https://github.com/openai/grade-school-math) — `main` config の `test` split を Hugging Face `datasets` 経由でロード。MIT ライセンス。
- **Sampling**: `random.Random(42).sample(test, 50)` で 50 問を固定抽出。各問題に `gsm8k_test_<idx>` の `question_id` を付与。
- **Gold answer 抽出**: 各 reference answer の末尾 `#### N` パターンから整数を抽出。
- **Model**: Azure OpenAI のチャットデプロイメント (実行時に `meta.json` にデプロイ名・API バージョンを記録)。

## 5. Variables

### Independent
- 軸1: Output Format × Prompt 条件 (5)
- 軸2: Step 数 (4 水準; `so_step_constrained` 内のサブ軸)
- 軸3: Runner (3)
- 軸4: Temperature set (2; `det` と `sc`)

### Dependent
- 正解率 (per-question correct の平均)
- Parse 失敗率
- 平均応答ステップ数 (改行 or List 長)
- 平均 prompt / completion tokens
- 平均レイテンシ (ms)

### Controlled
- プロンプト文言 (`src/sor_lab/prompts.py` で定数化、テストで diff 監視)
- 質問順序 (50 問の question_id 固定)
- `seed=42` (OpenAI 系のみ best-effort 決定性)

## 6. Procedure (Conditions)

### 軸1: Output Format × Prompt

| Condition ID | Output | System Prompt | User Prompt | 期待 CoT 効果 |
|---|---|---|---|---|
| `plain_direct` | text | "You are a math tutor. Answer with only the final integer, nothing else." | "{question}" | なし |
| `plain_cot` | text | "You are a math tutor. Think step by step, then write the final integer on the last line prefixed by 'Answer: '." | "{question}" | あり |
| `so_answer_first` | `AnswerFirst` JSON | "You are a math tutor. Return JSON matching the schema." | "{question}" | なし |
| `so_reasoning_first` | `ReasoningFirst` JSON | 同上 | "{question}" | あり |
| `so_step_constrained` | `StepConstrained` JSON | 同上 | "{question}" | あり (別軸で制御) |

### 軸2: Step 数 (`so_step_constrained` 内)

- `n_steps ∈ {1, 3, 5, 10}` × `answer_first ∈ {True, False}` の 8 サブ条件
- `reasoning_steps: list[str]` を `min_length=1, max_length=n_steps` で制約
- 初期実行では縮小して `n_steps ∈ {1, 5, 10}` × `answer_first=False` の 3 サブ条件

### 軸3: Runner

| Runner | パッケージ | 呼び出し |
|---|---|---|
| `openai_sdk` | `openai` (公式) | `openai.AzureOpenAI` |
| `azure_sdk`  | `azure-ai-inference` | `azure.ai.inference.ChatCompletionsClient` |
| `langchain`  | `langchain-openai` | `AzureChatOpenAI` + `with_structured_output` |

### 軸4: Temperature × N

| Set | temperature | N (試行) | 用途 |
|---|---|---|---|
| `det` | 0.0 | 1 | 決定的測定 (主指標) |
| `sc`  | 0.7 | 5 | Self-consistency |

### コール数 (初期実行)

- Plain (2 条件) × 3 Runner × (1 + 5) × 50 = **1,800**
- Structured 基本 (2 条件) × 3 Runner × (1 + 5) × 50 = **1,800**
- Structured Step 縮小 (3 サブ条件) × 3 Runner × (1 + 5) × 50 = **2,700**
- 合計 **6,300 calls**

## 7. Prompts

`src/sor_lab/prompts.py` を正本とする。本書ではコード生成時に自動転記。
変更履歴は git log で追跡する。

## 8. Schemas

`src/sor_lab/schemas.py` を正本とする。本書では Pydantic クラス定義と
`model_json_schema()` の `properties` キー順を併記して再現性を担保する。

## 9. Analysis Plan (要約 — 詳細は [analysis_plan.md](analysis_plan.md))

- **主分析**: H1 を `ReasoningFirst − AnswerFirst` の正解率差で評価。
  同一 50 問のペア比較なので **McNemar 検定** (`statsmodels.stats.contingency_tables.mcnemar`) を用いる。有意水準 α = 0.05。
- **副分析**: 各条件の正解率に **Wilson 95% CI**。H4 はマージン同等性 (TOST) 風に
  ±2pt 帯域内かを判定。H5 は多数決 vs 単発の差を Runner × 条件ごとに集計。
- **記述統計**: 平均ステップ数、平均トークン数、平均レイテンシを表で報告。

## 10. Reproducibility

### 実行コマンド

```bash
uv sync --frozen
uv run sor-lab smoke                          # 疎通確認
uv run sor-lab run --stage b1                 # 順序 + Plain ベースライン
uv run sor-lab run --stage b2                 # ステップ数効果
uv run sor-lab run --stage b3                 # temperature=0.7×5
uv run sor-lab report <run_id>                # summary.md 生成
```

### 必要環境変数 (`.env`)

```
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_DEPLOYMENT_NAME=<deployment>
```

### `meta.json` に記録する内容

- `run_id`, `git_commit`, `git_dirty`
- `model`, `deployment`, `api_version`
- `conditions_yaml_sha256`
- `python_version`, パッケージ pin 一覧
- 実行コマンドライン

### 結果 JSONL のカラム定義

[`docs/experiment_design.md` §6 ロジック] と同一スキーマ
(`run_id`, `condition_id`, `runner`, `temp_set`, `temperature`, `try_idx`,
`question_id`, `gold_answer`, `prompt`, `raw_response`, `raw_response_keys`,
`parsed_answer`, `parsed_reasoning`, `parsed_reasoning_step_count`,
`parse_failed`, `correct`, `latency_ms`, `prompt_tokens`, `completion_tokens`)。

### 依存ピン留め

`pyproject.toml` の `dependencies` を `==` で固定し、`uv.lock` をコミット。
CI は `uv sync --frozen` を要求する。初版時に解決された主要バージョン:

| Package | Version |
|---|---|
| Python | 3.12.4 |
| openai | 2.36.0 |
| azure-ai-inference | 1.0.0b9 |
| langchain-openai | 1.2.1 |
| langchain-core | 1.4.0 |
| pydantic | 2.13.4 |
| pydantic-settings | 2.14.1 |
| datasets | 4.8.5 |
| typer | 0.25.1 |
| pandas | 3.0.3 |
| python-dotenv | 1.2.2 |
| pyyaml | 6.0.3 |
| statsmodels | 0.14.6 |
| pytest | 9.0.3 |
| pytest-cov | 7.1.0 |
| ruff | 0.15.12 |
| mypy | 2.1.0 |

## 11. Results

Stage B 実行後に [`docs/results_template.md`](results_template.md) をコピーして
`docs/results_<run_id>.md` を作成し、本書 §11 からリンクする。

## 12. Threats to Validity

- **モデル更新**: Azure OpenAI のデプロイ ID が同名のまま裏側のモデルバージョンが
  更新される可能性。実行時に `meta.json` へ API レスポンスのモデルバージョンも
  記録する。
- **GSM8K の test set 漏洩**: 大規模モデルの事前学習に test set が混入している
  可能性は否定できない。本研究の主軸は条件間の **差分** なので絶対精度の解釈には
  慎重に。
- **50 問サンプル変動**: サンプル数が小さいので分散が大きい。McNemar 検定のペア
  設計でこれを部分的に緩和。
- **strict JSON Schema 仕様変更**: OpenAI / Azure 側で仕様が更新されると順序保証が
  揺れる可能性。実行ごとに `raw_response_keys` を assert して検出する。
- **Runner 間の挙動差**: `seed` パラメータの扱いが Runner で異なる場合がある。
  smoke test で挙動を確認する。

## 13. References

- Wei et al., 2022. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS.
- Nye et al., 2021. *Show Your Work: Scratchpads for Intermediate Computation with Language Models.*
- Wang et al., 2022. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR.
- Cobbe et al., 2021. *Training Verifiers to Solve Math Word Problems.* (GSM8K dataset paper)

## Revision History

- 2026-05-13: 初版 (実装前事前登録)。
