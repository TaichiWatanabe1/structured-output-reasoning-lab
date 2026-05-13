# Results — `<run_id>`

> 実験実行後に本テンプレを `docs/results_<run_id>.md` にコピーして埋める。

## Meta

- **run_id**: `<run_id>`
- **git_commit**: `<sha>` (git_dirty: `<true|false>`)
- **stages executed**: b1 / b2 / b3 / smoke のうち実行したものを明記
- **models**:
  - `gpt-4.1` = deployment `<name>`, model_string `<from API>`, api_version `<>`, reasoning_effort `null`
  - `gpt-4.1-mini` = deployment `<name>`, model_string `<from API>`, api_version `<>`, reasoning_effort `null`
  - `gpt-5` = deployment `<name>`, model_string `<from API>`, api_version `<>`, reasoning_effort `minimal`
- **EXEC_SEED**: `42`
- **conditions.yaml SHA256**: `<sha256>`
- **questions**: MGSM `ja`, n=250 (全件)
- **execution dates (UTC)**:
  - b1: `<ISO 8601 start>` 〜 `<end>`
  - b2: `<>`
  - b3: `<>`
- **total calls executed**: `<n>` / 108,000 (実行 stage 分のみ)
- **総コスト概算**: `<USD> / ¥<JPY>`

## 1. セル別主要指標表

各セル (Condition × Runner × Model × temp_set) に対し以下を表示:

- **acc**: 正解率 (%)。`sc` は単発平均 / 多数決の 2 値
- **CI95**: 95% Newcombe 単一比率 CI (paired は H1/H2 等の差分表で別途表示)
- **n**: 評価対象問題数 (Quality gate / Parse 失敗除外後)
- **parse_fail**: parse 失敗率 (%)
- **kov**: key_order_violation_rate (%) — `>1.0` の cell は H1 主検定対象外
- **steps**: 平均 `parsed_reasoning_step_count` (SO のみ)
- **tok_in / tok_out**: 平均 prompt / completion tokens
- **lat_ms**: 平均レイテンシ (ms)

### 1.1 `temp_set=det` × `gpt-4.1`

| Condition | Runner | acc [CI95] | n | parse_fail | kov | steps | tok_in / tok_out | lat_ms |
|---|---|---|---|---|---|---|---|---|
| `plain_direct` | openai_sdk |  |  |  |  | — |  |  |
| `plain_direct` | azure_sdk |  |  |  |  | — |  |  |
| `plain_direct` | langchain |  |  |  |  | — |  |  |
| `plain_cot` | openai_sdk |  |  |  |  | — |  |  |
| ... | ... |  |  |  |  |  |  |  |
| `so_answer_first` | openai_sdk |  |  |  |  |  |  |  |
| `so_reasoning_first` | openai_sdk |  |  |  |  |  |  |  |
| `so_step_constrained:n_steps=1` | openai_sdk |  |  |  |  |  |  |  |
| `so_step_constrained:n_steps=3` | openai_sdk |  |  |  |  |  |  |  |
| `so_step_constrained:n_steps=5` | openai_sdk |  |  |  |  |  |  |  |
| `so_step_constrained:n_steps=10` | openai_sdk |  |  |  |  |  |  |  |

(他 Runner も同形式で続ける)

### 1.2 `temp_set=det` × `gpt-4.1-mini`

(同上)

### 1.3 `temp_set=det` × `gpt-5` (reasoning_effort=minimal)

(同上)

### 1.4 `temp_set=sc` (T=0.7, 5 試行) × 3 model

各 model × condition × runner で:
- **acc_single**: 5 試行 × 250 問の単発正解率平均
- **acc_majority**: 多数決後の 250 問正解率
- Newcombe CI は acc_majority 側に付与
- `key_order_violation_rate` は 5 試行全体での集計

## 2. 仮説判定

### 2.1 H1: 順序効果 (主仮説)

#### Primary (1 cell): `gpt-4.1 × openai_sdk × det`

| 項目 | 値 |
|---|---|
| `so_answer_first` acc [CI95] | xx.x% [aa.a%, bb.b%] |
| `so_reasoning_first` acc [CI95] | xx.x% [aa.a%, bb.b%] |
| McNemar 2×2 table (a / b / c / d) | a / b / c / d |
| - b (Reasoning正解, Answer不正解) | b |
| - c (Reasoning不正解, Answer正解) | c |
| McNemar exact p | p |
| 正解率差 Δ = acc(R) − acc(A) | +xx.x pt |
| Newcombe paired CI95 | [aa.a, bb.b] pt |
| Quality gate (kov ≤ 1% for both conds) | pass / fail |

→ **H1 採択 / 棄却 / inconclusive**: 採択基準 (p < 0.05 かつ CI 下限 > 0 かつ Δ ≥ +5pt かつ QG pass) を全て満たしたかを明記。

#### Secondary (8 cell descriptive replication)

| Model | Runner | Δ 点推定 | Newcombe paired CI95 | 同方向? | kov |
|---|---|---|---|---|---|
| gpt-4.1 | azure_sdk |  |  |  |  |
| gpt-4.1 | langchain |  |  |  |  |
| gpt-4.1-mini | openai_sdk |  |  |  |  |
| gpt-4.1-mini | azure_sdk |  |  |  |  |
| gpt-4.1-mini | langchain |  |  |  |  |
| gpt-5 | openai_sdk |  |  |  |  |
| gpt-5 | azure_sdk |  |  |  |  |
| gpt-5 | langchain |  |  |  |  |

→ **記述**: 9 cell 中いくつで点推定が +5pt 以上か、CI が 0 を含むか、Quality gate を通ったか。

### 2.2 H2: CoT 等価性 (TOST ε=±3pt, α'=0.0028 Bonferroni 補正後)

| Model | Runner | Pair | Δ 点推定 | Newcombe CI | TOST p | CI ⊂ [-3, +3]? | 判定 |
|---|---|---|---|---|---|---|---|
| gpt-4.1 | openai_sdk | so_reasoning_first vs plain_cot |  |  |  |  |  |
| gpt-4.1 | openai_sdk | so_answer_first vs plain_direct |  |  |  |  |  |
| ... 残り 16 row |  |  |  |  |  |  |  |

→ **H2 採択 / 棄却**: 採択基準は TOST p < α' かつ CI 完全内包。primary 候補 (gpt-4.1 × openai_sdk) は α=0.05 でも別途報告。

### 2.3 H3: ステップ数効果 (per-point CI + Cochran-Armitage)

各 Model × Runner で `n_steps ∈ {1,3,5,10}` の正解率 + CI:

| Model | Runner | n_steps=1 [CI] | n_steps=3 [CI] | n_steps=5 [CI] | n_steps=10 [CI] | Cochran-Armitage z, p |
|---|---|---|---|---|---|---|
| gpt-4.1 | openai_sdk |  |  |  |  |  |
| ... |  |  |  |  |  |  |

→ **記述**: 曲線形状 (単調増加 / 飽和 / 山型) を Model × Runner 別に。Cochran-Armitage は副次。

### 2.4 H4: Runner 非依存性 (TOST ε=±5pt, α'=0.00139 Bonferroni 補正後)

| Model | Condition | Pair (Runner A vs B) | Δ | Newcombe CI | TOST p | CI ⊂ [-5, +5]? | 判定 |
|---|---|---|---|---|---|---|---|
| gpt-4.1 | plain_direct | openai_sdk vs azure_sdk |  |  |  |  |  |
| ... 残り 35 row |  |  |  |  |  |  |  |

→ **H4 採択 / 棄却**: 36 cell 中いくつ TOST pass か。pass しない条件は `raw_response_keys` diff を §3 で考察。

### 2.5 H5: Self-consistency

| Model | Condition | Runner | acc_det | acc_majority | Δ | Newcombe paired CI | 判定 |
|---|---|---|---|---|---|---|---|
| gpt-4.1 | plain_cot | openai_sdk |  |  |  |  |  |
| ... |  |  |  |  |  |  |  |

→ **H5 採択 / 棄却**: CoT 系で CI 下限 > 0 かつ Δ ≥ +2pt の cell をカウント。

### 2.6 H6: Model family 比較

#### 2.6a within-generation (gpt-4.1 vs gpt-4.1-mini)

| Runner | Δ_gpt-4.1 | Δ_gpt-4.1-mini | Δ差 = Δ_mini − Δ_4.1 | Newcombe paired CI | mini ≥ non-mini? |
|---|---|---|---|---|---|
| openai_sdk |  |  |  |  |  |
| azure_sdk  |  |  |  |  |  |
| langchain  |  |  |  |  |  |

→ **H6a 判定**: 3 Runner 中 2 以上で Δ差 > 0 かつ CI 下限 > 0 → 支持 / そうでない → 不支持。

#### 2.6b cross-architecture META (gpt-4.1 vs gpt-5 minimal)

| Runner | Δ_gpt-4.1 | Δ_gpt-5(min) | Δ差 = Δ_4.1 − Δ_5min | Newcombe paired CI | 縮小有意? (CI ∌ 0) | Δ差 ≥ +3pt? |
|---|---|---|---|---|---|---|
| openai_sdk |  |  |  |  |  |  |
| azure_sdk  |  |  |  |  |  |  |
| langchain  |  |  |  |  |  |  |

→ **H6b META 判定**: 3 Runner それぞれで 2 条件 (Δ差 ≥ +3pt かつ CI ∌ 0) を満たすかを評価。
解釈は「縮小が観測された」までに留め、「完全に消えた」とは主張しない。

### 2.7 H_QG: Quality gate

| Condition | Model | Runner | n | violations | rate (%) | 主検定対象? |
|---|---|---|---|---|---|---|
| (rate > 0.5% の cell のみ列挙) |  |  |  |  |  |  |

→ **除外 cell リスト**: rate > 1% で H1 主検定から外した cell を箇条書きで明示。

## 3. Parse 失敗・key_order_violation 詳細

- **Parse 失敗内訳**: 条件 × Model × Runner × 件数の表
- **key_order_violation 内訳**: 条件 × Model × Runner × 件数 + サンプル `raw_response_keys` 例

## 4. ランダム化監査

- **stage × quintile**: `run_order_idx` を 5 等分し、各 quintile × `condition_id` のクロス集計 + chi-square p
- **stage × quintile × Model**: 同上で Model についても
- **model_string 一致**: 各 model_key で stage 内・stage 間で `model_string` が一致したか (差分があった場合の時刻範囲)

→ **判定**: p > 0.05 (一様性が棄却されない) を確認。棄却された場合は該当 stage の交絡を §6 探索的所見で議論。

## 5. 探索的所見

(自由記述: 観察された興味深いパターン、想定外の挙動、フォローアップ案。
特に H6b META 仮説に関する gpt-5 の挙動、StepConstrained 曲線の形状、Runner 間の挙動差など)

## 6. 補足図表

(matplotlib PNG / Markdown 表へのリンク。例:
- StepConstrained 折れ線 PNG (Model × Runner)
- Model 間正解率ヒートマップ
- run_order_idx 分布)
