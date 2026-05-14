# Experiment Design — Structured Output Field Order and Reasoning Accuracy

> 本書は **データ収集前に確定**させる事前登録 (pre-registration) 文書です。
> コード変更が条件・分析手法に影響を与える場合は、本書を先に更新してください。

## 1. Abstract

大規模言語モデル (LLM) は左から右へ自己回帰的にトークンを生成するため、生成順序が
そのまま条件付け確率の連鎖を決定する (Chain-of-Thought; Wei et al., 2022)。
Pydantic / JSON Schema による Structured Output を用いる場合、フィールドの宣言順序が
そのまま生成順序となるため、`answer` を先に置くか `reasoning` を先に置くかで
最終解答の精度が大きく変わると予想される。

本研究では Azure OpenAI に対し、(i) Structured Output のフィールド順序、
(ii) reasoning ステップ数、(iii) クライアントライブラリ (OpenAI 公式 SDK /
Microsoft Azure AI Inference SDK / LangChain)、(iv) 温度設定、
(v) モデルデプロイメント (`gpt-4.1` / `gpt-4.1-mini` / `gpt-5` (`reasoning_effort=minimal`))
の 5 軸で MGSM-ja **全 250 問** の解答精度を測定し、上記仮説を検証する。
入出力言語は日本語に固定する (本番システムが日本語向けのため)。
副次的に、reasoning architecture を内蔵する `gpt-5` で順序効果が縮小するかを
**meta-hypothesis (H6b)** として検証する。言語間比較・`reasoning_effort` sweep・
追加モデルは §13 Future Work とする。

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
| H1 (primary) | `ReasoningFirst` の正解率は `AnswerFirst` を有意に上回る (想定 +5pt 以上)。**主検定 cell = `gpt-4.1 × openai_sdk × det`** (α=0.05、単一検定) | McNemar exact p 値、Newcombe paired CI、正解率差 |
| H1 (secondary) | 残り 8 cell (3 Model × 3 Runner − 1) で同方向の効果を **descriptive replication** として確認 | Newcombe paired CI、点推定 (有意性検定はしない) |
| H2 | `so_reasoning_first` ≈ `plain_cot` (ε=±3pt 同等)。`so_answer_first` ≈ `plain_direct` (ε=±3pt 同等) | TOST (proportions_ztost) と Newcombe paired CI が ε 帯域に完全内包 |
| H3 | JSON schema の `reasoning_steps: list[str]` の `max_length` でステップ上限を制約する**ステップ制約スキーマ設計 (step-constrained schema design)** において、上限 `n_steps ∈ {1, 3, 5, 10}` の増加に対して正解率が単調増加または飽和する。(これは単なる「ステップ数を増やす効果」ではなく、スキーマ設計の制約が生成質に影響するという仮説) | 各点の正解率 + Newcombe CI で曲線提示。Cochran-Armitage 傾向検定を副次 |
| H4 | `openai_sdk` / `azure_sdk` / `langchain` の 3 Runner 間で同条件の正解率差は ±5pt 以内 | proportions_ztost で形式 TOST、Newcombe CI |
| H5 | `temperature=0.7 × 5 試行` の多数決正解率は `temperature=0` を上回る (CoT 系で顕著) | 多数決 vs 単発正解率差、Newcombe CI |
| H6a | within-generation: `gpt-4.1` と `gpt-4.1-mini` の順序効果差。**mini ≥ non-mini** (小型ほど順序に依存) を支持するか | 順序効果差 `Δ_mini − Δ_4.1`、paired CI |
| H6b (META) | cross-architecture: `gpt-5` (`reasoning_effort=minimal`) の順序効果 `Δ_5min` は `gpt-4.1` の `Δ_4.1` より小さい。差 `Δ_4.1 − Δ_5min ≥ +3pt` かつ paired CI が 0 を含まない場合に「reasoning architecture が SO field 順序効果を抑制する」を支持 | 順序効果差、paired CI |

## 4. Materials

- **Dataset**: [MGSM-ja](https://huggingface.co/datasets/juletxara/mgsm) (Shi et al., 2022) の `ja` split を Hugging Face `datasets` 経由でロード。MGSM は GSM8K の人手翻訳版 (CC-BY-4.0)。
- **Sampling**: MGSM-ja の **全 250 問** を使用 (`n=250`)。`question_id = mgsm_ja_<idx>` (`idx ∈ {0..249}`)。
  - 部分抽出は行わない。検出力確保のため全件投入。
- **Gold answer 抽出**: MGSM の `answer_number` フィールドを整数として直接利用。
- **Models** (軸5; 3 個並行): `conditions.yaml` の `models:` セクションで `key` / `deployment` / `reasoning_effort` を宣言する。
  - `gpt-4.1`: 本番主力モデル相当。`reasoning_effort` は `null`。
  - `gpt-4.1-mini`: 同世代小型モデル (H6a 検証)。`reasoning_effort` は `null`。
  - `gpt-5`: `reasoning_effort=minimal` 固定 (hidden CoT トークン爆発を抑え純粋にアーキテクチャ差を測る)。`gpt-4.1` / `gpt-4.1-mini` には `reasoning_effort` を渡さない (パラメータ未対応のため)。H6b META 検証のためのモデル。
  - `deployment` はマシン固有の Azure デプロイ名で、各実行者が `conditions.yaml` を書き換える。`.env` には credentials のみを置く (旧 `AZURE_OPENAI_DEPLOYMENT_NAME_*` / `AZURE_OPENAI_REASONING_EFFORT` env var は廃止)。
  - 実行時に `meta.json` に各デプロイ名・API バージョン・モデルバージョン文字列 (API レスポンスの `model` フィールド)・`reasoning_effort` 値を記録

## 5. Variables

### Independent
- 軸1: Output Format × Prompt 条件 (7; うち 2 は `b0` 文献比較校正用、主仮説の検定対象外)
- 軸2: Step 数 (4 水準; `n_steps ∈ {1, 3, 5, 10}` × `answer_first=False` 固定の `so_step_constrained` サブ軸)
- 軸3: Runner (3; `openai_sdk` / `azure_sdk` / `langchain`)
- 軸4: Temperature set (2; `det` と `sc`)
- 軸5: **Model** (3; `gpt-4.1` / `gpt-4.1-mini` / `gpt-5` (`reasoning_effort=minimal` 固定))

言語軸 (en/ja) は本研究では扱わない (§13 Future Work)。

### Dependent
- 正解率 (per-question correct の平均)
- Parse 失敗率
- 平均応答ステップ数 (改行 or List 長)
- 平均 prompt / completion tokens
- 平均レイテンシ (ms)

### Controlled
- プロンプト文言 (`src/sor_lab/prompts.py` で日本語固定、テストで diff 監視)
- Pydantic field 名・description ともに固定 (`reasoning`, `answer`, `reasoning_steps` / 日本語 description)
- 質問順序 (250 問の `question_id` 固定)
- `seed`: `det` のみ `42` を渡す (best-effort 決定性)。`sc` (`n_tries=5`, `temperature=0.7`) では **seed 未指定** で SDK / API のランダムサンプリングに任せ、5 試行の独立サンプル性を確保する。`gpt-5` の `reasoning_effort` は `minimal` 固定
- **実行順ランダム化**: 各 stage 内の call list を `EXEC_SEED=42` で `random.Random(EXEC_SEED).shuffle(...)` する。これにより条件 × Model × Runner と実行時刻の交絡を除去する (§6 ランダム化プロトコル参照)

## 6. Procedure (Conditions)

### 軸1: Output Format × Prompt

| Condition ID | Output | System Prompt | User Prompt | 期待 CoT 効果 |
|---|---|---|---|---|
| `plain_direct` | text | "You are a math tutor. Answer with only the final integer, nothing else." | "{question}" | なし |
| `plain_cot` | text | "You are a math tutor. Think step by step, then write the final integer on the last line prefixed by '答え: '." | "{question}" | あり |
| `so_answer_first` | `AnswerFirst` JSON | "You are a math tutor. Return JSON matching the schema." | "{question}" | なし |
| `so_reasoning_first` | `ReasoningFirst` JSON | 同上 | "{question}" | あり |
| `so_step_constrained` | `StepConstrained` JSON | 同上 | "{question}" | あり (別軸で制御) |
| `lit_mgsm_direct_ja` | text | "問題を解いてください。整数の答えだけを出力してください。" | "{question}" | なし (`b0` 校正用) |
| `lit_mgsm_native_cot_ja` | text | "問題をステップごとに解き、最後に「答え: 数字」の形式で整数の答えを書いてください。" | "{question}" | あり (`b0` 校正用) |

### 軸2: Step 数 (`so_step_constrained` 内)

- `n_steps ∈ {1, 3, 5, 10}` × `answer_first=False` 固定の 4 サブ条件
- `reasoning_steps: list[str]` を `min_length=1, max_length=n_steps` で制約
- `answer_first=True` のバリエーションは本実験では扱わない (H1 主仮説は軸1 で検証済みのため)

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
| `sc`  | 0.7 | 5 | Budgeted self-consistency (Wang et al., 2022): n を固定した多数決 |

### 軸5: Model

`conditions.yaml` の `models:` セクションで 3 model を宣言する (deployment 名は各自書き換え):

| `key` | `deployment` (例) | `reasoning_effort` | 役割 |
|---|---|---|---|
| `gpt-4.1` | `<my-gpt-4.1-deploy>` | `null` | 本番主力モデル相当 (H1 primary cell) |
| `gpt-4.1-mini` | `<my-gpt-4.1-mini-deploy>` | `null` | 同世代小型モデル (H6a 検証) |
| `gpt-5` | `<my-gpt-5-deploy>` | `minimal` 固定 | reasoning architecture モデル (H6b META 検証) |

`api_version` は 3 モデルで揃える (`.env` の `AZURE_OPENAI_API_VERSION`)。`gpt-5` のみ `reasoning_effort=minimal` を渡す
(hidden CoT トークン爆発を抑え、純粋にアーキテクチャ差を見るため)。

### コール数

- Plain (2 cond) + SO 基本 (2 cond) + StepConstrained (4 sub-cond) = **8 cond** (主仮説対象)
- `b0` 文献比較校正: **2 cond** (`lit_mgsm_direct_ja` / `lit_mgsm_native_cot_ja`、記述統計のみ)
- 8 cond × 3 Runner × (1 + 5) try × 250 Q × 3 Model = **108,000 calls** (主実験)

stage 内訳:
- `b0` = 文献比較校正 (2 lit 条件) × 3 Runner × 1 (det) × 250 Q × 3 Model = **4,500 calls**
- `b1` = Plain + SO 基本 (4 cond) × 3 Runner × 1 (det) × 250 × 3 Model = **9,000 calls**
- `b2` = StepConstrained (4 sub-cond) × 3 Runner × 1 (det) × 250 × 3 Model = **9,000 calls**
- `b3` = 全 8 cond × 3 Runner × 5 (sc) × 250 × 3 Model = **90,000 calls**
- 合計 **112,500 calls**

予算試算 (Azure Global Standard、`gpt-5` は `reasoning_effort=minimal`):

| Model | Phase 全体コスト | 円 (¥150/USD) |
|---|---|---|
| `gpt-4.1` | $102 | ¥15,300 |
| `gpt-4.1-mini` | $20.5 | ¥3,100 |
| `gpt-5` (minimal) | $119 | ¥17,800 |
| **計** | **~$242** | **~¥36,300** (+15% で **~¥41,700**) |

### ランダム化プロトコル

各 stage 内で、その stage が含む全コールを `(condition_id, runner, temp_set, try_idx, model_key, question_id)`
のタプル列に展開し、`random.Random(EXEC_SEED=42).shuffle(...)` で順序を確定する。
シャッフル後の連番を JSONL の `run_order_idx` カラムに記録する。
- stage 間 (b1 ↔ b2 ↔ b3) のシャッフルは行わない (作業単位を保持するため)。stage 間交絡は §12 Threats で明示する。
- API 失敗時の再試行は元の `run_order_idx` を保持したまま append-only で書き込む (skip キーは下記 §10 を参照)。

## 7. Prompts

`src/sor_lab/prompts.py` を正本とし、全文と固定方針を [`prompt_canonical.md`](prompt_canonical.md) §1 に Appendix 化する (NeurIPS / ICLR / Responsible NLP checklist 準拠)。日本語固定 (`PROMPTS[condition_id]` の単一辞書)。

**条件間の対称性 (固定条件、コードと `tests/test_prompts.py` で監視):**
- `so_answer_first` / `so_reasoning_first` / `so_step_constrained` の **system prompt は完全に同一文字列**。
- `so_answer_first` と `so_reasoning_first` の差分は **Pydantic schema の field 宣言順のみ** (`description` も含めて他は完全一致)。
- `plain_direct` と `plain_cot` の差分は CoT 指示の有無のみ。`plain_cot` の answer marker は日本語ネイティブ `「答え: 」` (英語 `Answer:` ではない; MGSM 原論文との整合性・言語交絡排除のため 2026-05-14 変更)。

変更履歴は git log で追跡し、`tests/test_prompts.py` の SHA-256 digest assert で diff を CI 監視する。

## 8. Schemas

`src/sor_lab/schemas.py` を正本とし、全文と field/description 文言を [`prompt_canonical.md`](prompt_canonical.md) §2 に Appendix 化する。**field 名は英語固定**
(`reasoning`, `answer`, `reasoning_steps`)、`Field(description=...)` は日本語固定。
`model_json_schema()` の `properties` キー順が宣言順と一致することを
`tests/test_schemas.py` で assert する。

### Runtime schema validation

各 Runner は API レスポンスを `schema_cls.model_validate(data)` に通してから採点する。
`ValidationError` (answer 欠落・型違い、`extra="forbid"` 違反、`reasoning_steps` の
`min_length` / `max_length` 違反など) は `parse_failed=True` として **denominator から除外**。
これにより `answer` だけ正しい不正な構造の応答が偽陽性として `correct=True` になることを防ぐ。

### Runtime quality gate (key order)

ランタイムで各 API 応答の `raw_response_keys` を取得し、宣言フィールド順と一致しなければ
`key_order_violation=True` を JSONL 行に立てる。`raw_response_keys` は schema 検証の **前** に
`json.loads(...).keys()` の挿入順で取得する (= モデル生成順)。SO 両条件 (`so_answer_first` /
`so_reasoning_first`) のいずれかで cell 単位の違反率が **> 1%** の場合、その cell は
H1 主検定 (McNemar exact + Newcombe paired CI) の対象から **自動除外** され、
`summary.md` の H1 表で `excluded_by_key_order_gate=True` / `mcnemar_p=NaN` として明示される。
JSON Schema 仕様自体は property 順序を意味論として保証しないため、API/SDK/モデル
バージョンの変化を検出するためのランタイムゲートとして必須。

## 9. Analysis Plan (要約 — 詳細は [analysis_plan.md](analysis_plan.md))

- **H1 (primary)**: `gpt-4.1 × openai_sdk × det` で `so_reasoning_first` − `so_answer_first` の
  ペア比較を **McNemar exact** (`statsmodels.stats.contingency_tables.mcnemar`, `exact=True`)
  で 1 検定。α=0.05、採択基準は p < 0.05 かつ Newcombe paired diff CI の下限 > 0
  かつ点推定 ≥ +5pt。残り 8 cell は descriptive replication として **Newcombe CI のみ報告**。
- **H2 / H4 同等性**: `statsmodels.stats.proportion.proportions_ztost` で形式 TOST。
  ε は H2=±3pt、H4=±5pt。同時に Newcombe paired CI を提示し、CI が ε 内に完全内包
  されていることを採択条件とする。
- **H3 step 数**: 各 `n_steps` 点の正解率と Newcombe CI を Model × Runner ごとに表示し、
  曲線として議論。Cochran-Armitage 傾向検定を副次として併記。Spearman は採用しない。
- **H5 budgeted self-consistency**: 多数決 vs 単発の正解率差を Model × Runner × 条件ごとに集計。`sc` = budgeted SC (Wang et al., 2022): `T=0.7, n=5` の固定 budget 多数決。
- **H6 model 比較**: (a) `gpt-4.1` vs `gpt-4.1-mini` の順序効果差、(b) `gpt-4.1` vs `gpt-5(minimal)`
  の順序効果差 (META)。いずれも paired CI ベースで判定。
- **Wilson CI** は単一比率の表示専用 (paired diff には使用しない)。
- **記述統計**: 平均ステップ数、prompt/completion トークン数、レイテンシ、`key_order_violation_rate`
  を Model × Runner × 条件ごとに表で報告。
- **ランダム化監査**: `run_order_idx` と `condition_id` / `model_key` を chi-square で
  クロス集計し、実行順偏りの不在を確認 (§3 補助分析)。

## 10. Reproducibility

### 実行コマンド

```bash
uv sync --frozen
uv run sor-lab smoke                  # 疎通確認 (3 model × 主要4条件 × 1 Q = 12 calls)

uv run sor-lab run --stage b1         # Plain + SO 基本 (4 cond) × det × 3 model
uv run sor-lab run --stage b2         # StepConstrained (4 sub-cond) × det × 3 model
uv run sor-lab run --stage b3         # 全 8 cond × sc (T=0.7×5) × 3 model

uv run sor-lab report <run_id>        # summary.md 生成
```

### 必要環境変数 (`.env`)

`.env` は **credentials と環境依存値のみ**。実験パラメータ (model 軸を含む) は `conditions.yaml` に集約する。

```
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
AZURE_OPENAI_API_VERSION=2025-04-01-preview
EXEC_SEED=42                            # ランダム化シード (省略時 42)
```

旧 `AZURE_OPENAI_DEPLOYMENT_NAME` / `AZURE_OPENAI_DEPLOYMENT_NAME_GPT4O*` /
`AZURE_OPENAI_DEPLOYMENT_NAME_GPT41[_MINI]` / `AZURE_OPENAI_DEPLOYMENT_NAME_GPT5` /
`AZURE_OPENAI_REASONING_EFFORT` の後方互換サポートは行わない。これらは
`conditions.yaml` の `models:` セクション (`key` / `deployment` / `reasoning_effort`)
に置き換わった。

### `meta.json` に記録する内容

- `run_id`, `git_commit`, `git_dirty`
- `stage` (`b1` / `b2` / `b3` / `smoke`)
- `models`: `[{"key": "gpt-4.1", "deployment": "<name>", "api_version": "...", "model_string": "<API レスポンス由来>", "reasoning_effort": null}, {"key": "gpt-4.1-mini", ...}, {"key": "gpt-5", "reasoning_effort": "minimal", ...}]`
- `exec_seed` (= `EXEC_SEED`)
- `conditions_yaml_sha256`
- `python_version`, パッケージ pin 一覧
- 実行コマンドライン

### 結果 JSONL のカラム定義

```
run_id, condition_id, runner, temp_set, temperature, try_idx,
question_id, model_key (gpt-4.1 / gpt-4.1-mini / gpt-5),
model_string (API レスポンスの model フィールド),
deployment, reasoning_effort (minimal / null),
gold_answer, prompt, raw_response, raw_response_keys,
parsed_answer, parsed_reasoning, parsed_reasoning_step_count,
parse_failed, correct, key_order_violation,
latency_ms, prompt_tokens, completion_tokens,
run_order_idx
```

skip キーは `(run_id, condition_id, runner, temp_set, try_idx, model_key, question_id)`。
再実行時に同一キーの行があれば skip して append-only を維持する。

### `errors.jsonl` (失敗 call のログ)

Runner 例外時には `raw.jsonl` には書かず、`results/runs/<run_id>/errors.jsonl` に
`{run_id, condition_id, runner, model_key, temp_set, try_idx, question_id, run_order_idx, error, timestamp}`
を append-only で書き出す。`errors.jsonl` の行は集計の denominator には含めず、
`summary.md` 冒頭に件数とエラーレートを表示する (失敗が見えない過大評価を防ぐ)。
re-run 時は `raw.jsonl` のキーのみを skip 対象とし、エラー行は再試行可能。

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
  更新される可能性。実行時に `meta.json` および各 JSONL 行へ API レスポンスの `model`
  文字列を記録する。
- **GSM8K / MGSM の test set 漏洩**: 大規模モデルの事前学習に GSM8K test set が
  混入している可能性は否定できない。MGSM は人手翻訳された派生物だが、翻訳元の
  問題自体は GSM8K と同一。本研究の主軸は条件間の **差分** なので絶対精度の解釈には
  慎重に。
- **n=250 でも検出力に上限**: McNemar exact + α=0.05 で +5pt 差を検出するには
  ~250 ペアでも b−c ≈ 12 程度の偏りが必要。primary cell を 1 個に絞り、残り 8 cell は
  descriptive 扱いとすることで多重検定の検出力低下を回避する。
- **strict JSON Schema 仕様変更**: OpenAI / Azure 側で仕様が更新されると property 順
  保証が揺れる可能性。runtime quality gate (`key_order_violation` フラグ + 1% 閾値で
  H1 除外) で検出。JSON Schema 仕様自体は property 順序を semantic に保証しないため、
  これは品質ゲートとして必須。
- **Runner 間の挙動差**: `seed` パラメータの扱いが Runner で異なる場合がある。
  smoke test と H4 (Runner 同等性 TOST) で挙動を検査。
- **reasoning model の hidden CoT による順序効果マスキング (META 仮説の対象)**:
  `gpt-5` は `reasoning_effort=minimal` でも内部で何らかの reasoning token を生成
  しうる。H6b META 仮説で「順序効果が縮小していること」を示すが、これは
  「順序効果が完全に消えた」とは異なる解釈に留める。`reasoning_effort` を上げると
  さらに縮小するかは §13 Future Work で扱う。
- **stage 間の時間的交絡**: b1/b2/b3 を逐次実行するため、stage 単位ではモデル更新の
  影響が条件と相関する可能性がある。stage 内ではランダム化済 (`EXEC_SEED`)、stage 間
  比較を行う仮説 (H5: det vs sc) では、`model_string` 一致を集計時に確認することで
  影響を限定する。

## 13. Future Work

本実験で扱わない以下のテーマは、別 stage または独立論文で扱う:

1. **言語ローカライズ (旧 Phase C, L2a〜L2d)**: 質問 (Q) / プロンプト (P) /
   description (D) の言語を `(en, en, en)` から `(ja, ja, ja)` まで段階的に変える
   4 サブ条件で正解率を測る。description 言語のみの効果 (L2c vs L2d) は
   `主要4条件 × 2 lang × 1 model (mini) × 3 R × 6 try × 250 ≈ 36,000 calls` で
   ¥3,000 程度で実施可能。
2. **`reasoning_effort` sweep**: `gpt-5` の `reasoning_effort ∈ {minimal, low, medium}`
   で順序効果がどう変化するか。コスト爆発に注意 (`medium` は output 3-5x)。
3. **後続モデル**: `gpt-5.1+`, `o3-mini` 等のリリース動向に応じて追加検証。
4. **EN ベースライン**: GSM8K-en + 完全英語 prompt で文献値とのキャリブレーション (JA ベースラインは `b0_lit` ステージで対応済み、§6 参照)。
5. **`answer_first=True` × StepConstrained**: 本実験では `False` 固定としたが、
   step 数 × order の交互作用を見るには両側必要。

## 14. References

- Wei et al., 2022. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS.
- Nye et al., 2021. *Show Your Work: Scratchpads for Intermediate Computation with Language Models.*
- Wang et al., 2022. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR.
- Kojima et al., 2022. *Large Language Models are Zero-Shot Reasoners.* NeurIPS.
- Shi et al., 2022. *Language Models are Multilingual Chain-of-Thought Reasoners.* ICML. (MGSM dataset)
- Cobbe et al., 2021. *Training Verifiers to Solve Math Word Problems.* (GSM8K dataset paper)

## Revision History

- 2026-05-13: 初版 (実装前事前登録)。
- 2026-05-13: Phase B/C 構成へ拡張。軸5 (Model: gpt-4o / gpt-4o-mini) と軸6 (Language: L2a〜L2d) を追加。Phase B の主言語を `ja` 固定 (= L2d) とし、Phase C で段階的ローカライズ効果 (H7, H8) を検証する。Materials を GSM8K (en) から MGSM (ja+en) に変更。JSONL カラムに `phase` / `model_key` / `lang_sub` / `q_lang` / `p_lang` / `d_lang` を追加。
- 2026-05-13: 統計レビュー (n=50 検出力不足、paired CI/TOST 未定義、Spearman 弱小、ランダム化欠如、quality gate 不在) と予算制約を踏まえて最終改訂。**Phase C 全削除** (§13 Future Work に移動)、**Model 軸を `gpt-4.1` / `gpt-4.1-mini` / `gpt-5` (`reasoning_effort=minimal`) の 3 個に置換**、**n=50 → n=250 (MGSM-ja 全件)**、軸2 を full `{1,3,5,10}` に統一、H1 を primary cell (`gpt-4.1 × openai_sdk × det`) 単一検定 + secondary descriptive に再校正、H2/H4 を Newcombe + TOST に、H3 を per-point CI + Cochran-Armitage に、H6 を within-gen + cross-arch META に再定義、H7/H8 削除。実行順 `EXEC_SEED` ランダム化と `key_order_violation` quality gate (1% threshold) を導入。JSONL カラムから `phase`/`lang_sub`/`q_lang`/`p_lang`/`d_lang` を削除し `reasoning_effort` / `key_order_violation` / `run_order_idx` を追加。env vars を 3 model 用に置換 (後方互換削除)。
- 2026-05-14: データ収集前レビュー反映。(a) `sc` 試行の seed を未指定に変更 (5 試行の独立サンプル性確保)、(b) Runner 例外を `errors.jsonl` に永続化し `summary.md` に件数表示、(c) LangChain で `parsed.model_dump_json()` で raw を上書きしていた問題を修正 (`raw_response_keys` がモデル実生成順を保つよう、AIMessage.content をそのまま渡す)、(d) 全 Runner で `schema_cls.model_validate` を通った row のみ採点 (`ValidationError` は `parse_failed=True`)、(e) `order_effect_mcnemar` に `excluded_by_key_order_gate` フラグを実装し H1 主検定で 1% threshold gate を強制反映。Prompt / Schema canonical text を [`prompt_canonical.md`](prompt_canonical.md) に Appendix 化。
- 2026-05-14: 軸5 (Model: `key` / `deployment` / `reasoning_effort`) を `conditions.yaml` の `models:` セクションに集約。`.env` を credentials のみに縮退し、`AZURE_OPENAI_DEPLOYMENT_NAME_GPT41[_MINI]` / `AZURE_OPENAI_DEPLOYMENT_NAME_GPT5` / `AZURE_OPENAI_REASONING_EFFORT` env vars を **廃止** (後方互換なし)。`ExperimentConfig.model_specs` を新設し、`run_stage` は `cfg.model_specs` を参照する。これにより `conditions_yaml_sha256` がモデル軸を含む真の実験設定ハッシュになる。
- 2026-05-14: 文献比較対応。(a) `plain_cot` の answer marker を英語 `Answer:` から日本語 `答え:` に変更 (言語整合性・MGSM 原論文のプロンプトとの整合; `evaluation.py` の `_ANSWER_PREFIX_RE` も更新)、(b) `b0_lit` 文献比較校正ステージ追加 (`lit_mgsm_direct_ja` / `lit_mgsm_native_cot_ja` 2 条件 × 3 Runner × det × 250 Q × 3 Model = 4,500 calls、記述統計のみ、主仮説検定対象外)、(c) `sc` を「Budgeted self-consistency (Wang et al., 2022)」として明示化、(d) H3 を「ステップ制約スキーマ設計 (step-constrained schema design)」として正確にフレーミング。参照追加: Shi et al. 2022 (MGSM), Kojima et al. 2022。
