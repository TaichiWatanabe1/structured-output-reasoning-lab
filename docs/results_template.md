# Results — `<run_id>`

> Stage B 実行後に本テンプレを `docs/results_<run_id>.md` にコピーして埋める。

## Meta

- **run_id**: `<run_id>`
- **git_commit**: `<sha>`
- **model / deployment**: `<model name>` / `<deployment name>`
- **api_version**: `<api version>`
- **conditions.yaml SHA256**: `<sha256>`
- **stages executed**: B-1 / B-2 / B-3 のうち実行したものを明記
- **questions**: GSM8K test split, seed=42, n=50
- **execution date (UTC)**: `<ISO 8601>`

## 1. 主要指標表

### `temp_set=det` (temperature=0.0, 1 試行)

| Condition | openai_sdk | azure_sdk | langchain |
|---|---|---|---|
| `plain_direct`        | xx.x% [CI] | xx.x% [CI] | xx.x% [CI] |
| `plain_cot`           |            |            |            |
| `so_answer_first`     |            |            |            |
| `so_reasoning_first`  |            |            |            |
| `so_step_constrained:n_steps=1`  |            |            |            |
| `so_step_constrained:n_steps=5`  |            |            |            |
| `so_step_constrained:n_steps=10` |            |            |            |

### `temp_set=sc` (temperature=0.7, 5 試行)

| Condition | openai_sdk 多数決 | azure_sdk 多数決 | langchain 多数決 |
|---|---|---|---|
| ... | ... | ... | ... |

## 2. 仮説判定

### H1: 順序効果 (主仮説)

| Runner | so_answer_first | so_reasoning_first | 差分 | McNemar p | 判定 |
|---|---|---|---|---|---|
| openai_sdk |  |  |  |  |  |
| azure_sdk  |  |  |  |  |  |
| langchain  |  |  |  |  |  |

→ **H1 採択 / 棄却**: ...

### H2: CoT 等価性

- `so_reasoning_first` vs `plain_cot`: ...
- `so_answer_first` vs `plain_direct`: ...

→ **H2 採択 / 棄却**: ...

### H3: ステップ数効果

`n_steps` vs 正解率 (Runner ごとの Spearman ρ, p):

| Runner | ρ | p | 判定 |
|---|---|---|---|

→ **H3 採択 / 棄却**: ...

### H4: ライブラリ非依存性

| Condition | max 差 (3 Runner) | 判定 (≤ 2pt) |
|---|---|---|

→ **H4 採択 / 棄却**: ...

### H5: 安定性 (self-consistency)

| Condition × Runner | det | sc 多数決 | 差 | 判定 |
|---|---|---|---|---|

→ **H5 採択 / 棄却**: ...

## 3. Parse 失敗・キー順 warning

- Parse 失敗: 条件 × Runner × 件数
- raw_response_keys が宣言順と一致しなかった件数: ...

## 4. 探索的所見

(自由記述: 観察された興味深いパターン、想定外の挙動、フォローアップ案)

## 5. 補足図表

(matplotlib PNG / Markdown 表へのリンク)
