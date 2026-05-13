# Pre-registered Analysis Plan

> データ収集前に確定する分析計画。後付けで p-hacking を行わないための事前登録。
> 変更時は git log で履歴を残し、Stage B 実行前に必ずレビューする。

## 0. データの単位

JSONL の 1 行 = `(condition_id, runner, temp_set, try_idx, question_id)` の組に対する
1 コール結果。`temp_set=det` は `try_idx=0` のみ、`sc` は `try_idx ∈ {0..4}`。

## 1. 主要指標

### 1.1 正解率 (accuracy)

```
accuracy(c, r, t) = mean_over_questions(correct(c, r, t, ·, ·))
```

`temp_set=det` (1 試行) はそのまま、`temp_set=sc` (5 試行) は **多数決後の正解率** と
**単発正解率の平均** の 2 種類を別カラムで集計する。

### 1.2 Parse 失敗率

```
parse_fail_rate(c, r, t) = mean(parse_failed)
```

Parse 失敗は `correct=False` 扱いだが、別カラムで内訳を表示する。

### 1.3 平均ステップ数

`parsed_reasoning_step_count` の平均。Structured 系のみ算出。

### 1.4 平均レイテンシ・トークン

`latency_ms`, `prompt_tokens`, `completion_tokens` の平均。

## 2. 仮説検定

### H1: 順序効果 (主仮説)

- **比較**: `so_reasoning_first` vs `so_answer_first`、Runner ごとに独立に検定
- **データ**: `temp_set=det`、50 問のペア
- **検定**: McNemar 検定 (`statsmodels.stats.contingency_tables.mcnemar`、`exact=True`)
- **有意水準**: α = 0.05 (3 Runner で実施するので Bonferroni 補正後 α' = 0.0167)
- **効果量**: 正解率差 + 95% Wilson CI
- **判定**: p < α' かつ 正解率差 ≥ +5pt なら H1 採択

### H2: CoT 等価性

- **比較**: `so_reasoning_first` vs `plain_cot`、および `so_answer_first` vs `plain_direct`
- **データ**: `temp_set=det`、Runner ごと
- **判定**: 正解率差の 95% CI が ±3pt 帯域に収まれば H2 採択 (TOST 風)

### H3: ステップ数効果

- **比較**: `so_step_constrained` の `n_steps ∈ {1, 5, 10}` で正解率
- **データ**: `temp_set=det`、`answer_first=False`、Runner ごと
- **検定**: Spearman 順位相関 (`scipy.stats.spearmanr`) で `n_steps` と正解率の相関
- **判定**: ρ > 0 かつ p < 0.05 で単調増加と判定。飽和は plot で目視確認

### H4: ライブラリ非依存性

- **比較**: 同一条件で 3 Runner の正解率を比較
- **対象条件**: `plain_direct`, `plain_cot`, `so_answer_first`, `so_reasoning_first`
- **判定**: `max(3 Runner) − min(3 Runner) ≤ 2pt` で H4 採択。差が大きい条件は
  schema 変換層と raw_response_keys の diff を `summary.md` に列挙

### H5: 安定性 (self-consistency)

- **比較**: 同一条件 × Runner で `temp_set=sc` の多数決 vs `temp_set=det` の正解率
- **判定**: 多数決 ≥ det かつ CoT 系 (`plain_cot`, `so_reasoning_first`,
  `so_step_constrained`) で正解率差が +2pt 以上で H5 採択

## 3. 補助分析 (探索的)

- 条件 × question_id ヒートマップで「どの問題がどの条件で解けるか」可視化
- `parsed_reasoning_step_count` と `correct` のクロス集計 (条件 × ステップ数 × 正解率)
- Runner 間で同じ問題に対する `raw_response_keys` の順序差検出 (warning フラグ集計)

## 4. 出力

`sor-lab report <run_id>` が以下を `results/runs/<run_id>/summary.md` に出力:

1. メタ情報 (model, api_version, 50 問の question_id 一覧、git commit)
2. **§2.1–2.5 の検定結果テーブル** (条件 × Runner × 指標)
3. ステップ数 × 正解率 折れ線 (Markdown 表 + 任意 PNG)
4. 仮説判定サマリ (H1–H5 採択/棄却)
5. Parse 失敗・キー順序 warning の内訳
6. 探索的所見 (自由記述用テンプレ)

## 5. 事前登録の取り扱い

- 本書の §2 の検定設計はデータ収集前に確定。
- §3 の探索的分析は事後追加可能だが、§2 の主結論には影響させない。
- 検定の追加・変更は git log と `Revision History` に記録。

## Revision History

- 2026-05-13: 初版。
