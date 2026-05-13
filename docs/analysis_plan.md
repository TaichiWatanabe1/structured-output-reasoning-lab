# Pre-registered Analysis Plan

> データ収集前に確定する分析計画。後付けで p-hacking を行わないための事前登録。
> 変更時は git log で履歴を残し、stage 実行前に必ずレビューする。

## 0. データの単位

JSONL の 1 行 = `(condition_id, runner, temp_set, try_idx, model_key, question_id)`
の組に対する 1 コール結果。`temp_set=det` は `try_idx=0` のみ、`sc` は `try_idx ∈ {0..4}`。
言語は日本語固定 (本実験では言語軸を扱わない)。
各行に `run_order_idx` (stage 内シャッフル後の連番) と `key_order_violation` フラグを持つ。

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

#### Primary (主検定、1 cell)

- **比較**: `so_reasoning_first` vs `so_answer_first`
- **Cell**: `model_key=gpt-4.1` × `runner=openai_sdk` × `temp_set=det`
- **データ**: 250 問のペア
- **検定**: McNemar exact (`statsmodels.stats.contingency_tables.mcnemar`、`exact=True`)
- **有意水準**: α = 0.05 (単一検定、多重補正なし)
- **効果量**: 正解率差 + 95% **Newcombe paired CI** (method 10、`src/sor_lab/stats.py` に実装予定)
- **判定**: すべて満たした場合に H1 採択
  - McNemar exact p < 0.05
  - Newcombe paired diff CI の下限 > 0
  - 点推定 ≥ +5pt
- **Quality gate**: primary cell の対象 2 条件いずれかで `key_order_violation_rate > 1%` の場合は
  H1 主検定は **inconclusive** として報告し、`summary.md` に違反内訳を明示する。

#### Secondary (descriptive replication、残り 8 cell)

- **対象**: 3 Model × 3 Runner − 1 (primary) = 8 cell
- **検定**: 行わない。**Newcombe paired CI と点推定のみ報告**
- **目的**: primary cell の効果が他 cell でも同方向に出るかを記述的に確認

### H2: CoT 等価性

- **比較**: `so_reasoning_first` vs `plain_cot`、および `so_answer_first` vs `plain_direct`
- **データ**: `temp_set=det`、Model × Runner ごと
- **検定**: 形式 TOST (`statsmodels.stats.proportion.proportions_ztost`、ε=±3pt、α=0.05)
  + 95% Newcombe paired CI を併記
- **判定**: TOST 両側 p < 0.05 **かつ** Newcombe CI が ε=[-3pt, +3pt] 帯域に完全内包される場合に H2 採択
- **多重補正**: 3 Model × 3 Runner × 2 ペア = 18 セル。判定は Bonferroni 補正後 α' = 0.0028 で
  個別に行う。primary 候補 (gpt-4.1 × openai_sdk) は補正なし α = 0.05 で別途報告

### H3: ステップ数効果

- **比較**: `so_step_constrained` の `n_steps ∈ {1, 3, 5, 10}` (4 点) で正解率
- **データ**: `temp_set=det`、`answer_first=False`、Model × Runner ごと
- **主分析**: 各 `n_steps` 点の正解率と 95% Newcombe CI を Model × Runner で表示。
  曲線として記述 (Markdown 表 + 任意の PNG)。
- **副次検定**: Cochran-Armitage 傾向検定 (`statsmodels.stats.contingency_tables` のテーブル
  関数で 2 × 4 cross-tab を構築し z-trend) で p 値を併記
- **判定**: 「単調増加または飽和」を CI 曲線で議論。傾向検定の有意性は副次情報として
  扱う (n=250 × 4 点では Cochran-Armitage の検出力にも限界がある)
- **Spearman は採用しない** (4 点では検出力が極端に低いため)

### H4: ライブラリ非依存性

- **比較**: 同一条件 × Model で 3 Runner の正解率の pairwise 同等性
- **対象条件**: `plain_direct`, `plain_cot`, `so_answer_first`, `so_reasoning_first` (主要 4 条件)
- **検定**: 各 Model × 条件で 3 Runner の 3 ペア (`openai_sdk` vs `azure_sdk`,
  `openai_sdk` vs `langchain`, `azure_sdk` vs `langchain`) に対して
  `proportions_ztost` を ε=±5pt, α=0.05 で実施 + Newcombe paired CI 併記
- **判定**: TOST p < 0.05 **かつ** Newcombe CI が [-5pt, +5pt] 内包される場合に
  そのペアで「Runner 同等」を採択
- **多重補正**: 3 Model × 4 条件 × 3 ペア = 36 セル。Bonferroni 補正後 α' = 0.00139
- **副次**: 差が ε を超える条件は schema 変換層と `raw_response_keys` の diff を
  `summary.md` に列挙し、Runner 別の挙動差を質的に解説

### H5: 安定性 (self-consistency)

- **比較**: 同一条件 × Model × Runner で `temp_set=sc` の多数決正解率 vs `temp_set=det` の正解率
- **データ**: 主要4条件 (Plain + SO 基本) と StepConstrained (n_steps=5 を代表点とする)
- **指標**: 多数決正解率 − det 正解率 + Newcombe paired CI
- **判定**: CoT 系 (`plain_cot`, `so_reasoning_first`, `so_step_constrained:n_steps=5`) で
  Newcombe paired CI の下限 > 0 かつ点推定 ≥ +2pt なら該当条件で H5 採択

### H6: Model family 比較 (within-generation + cross-architecture META)

#### H6a: within-generation (gpt-4.1 vs gpt-4.1-mini)

- **比較**: H1 の順序効果差 `Δ = acc(so_reasoning_first) − acc(so_answer_first)` を 2 model 間で比較
- **データ**: `temp_set=det`、3 Runner で独立に算出
- **判定**: 各 Runner で `Δ_mini − Δ_4.1` の点推定と paired CI を表示。
  3 Runner 中 2 以上で点推定 > 0 かつ CI 下限 > 0 なら「mini ≥ non-mini (小型ほど順序に依存)」を支持

#### H6b: cross-architecture META (gpt-4.1 vs gpt-5(minimal))

- **比較**: H1 の順序効果差 `Δ` を `gpt-4.1` と `gpt-5` (reasoning_effort=minimal) で比較
- **データ**: `temp_set=det`、3 Runner で独立に算出
- **判定**: 3 Runner それぞれで以下を満たせば「reasoning architecture が SO field 順序効果を抑制」を支持
  - 点推定 `Δ_4.1 − Δ_5min ≥ +3pt`
  - 95% Newcombe paired CI が 0 を含まない (= 縮小が有意)
- **解釈の限定**: `reasoning_effort=minimal` でも内部 reasoning token が完全に 0 とは限らないため、
  「順序効果が完全に消えた」とは異なる。「縮小が観測された」までの主張に留める

### H7, H8 (削除)

言語ローカライズ関連の仮説は本実験スコープ外 ([experiment_design.md §13 Future Work](experiment_design.md#13-future-work) 参照)。

### H_QG: Quality gate (新規)

- **判定**: 任意の条件 × Model × Runner で `key_order_violation_rate > 1%` の場合、
  その cell は H1〜H6 の主検定対象から除外し、`summary.md` の Quality gate セクションに違反
  内訳 (条件 × Model × Runner × `key_order_violation_rate`) を明示する。
- **目的**: strict JSON Schema の property 順序保証が崩れた状態で順序効果を評価することを防ぐ。

## 3. 補助分析 (探索的)

- 条件 × question_id ヒートマップで「どの問題がどの条件で解けるか」可視化
- `parsed_reasoning_step_count` と `correct` のクロス集計 (条件 × ステップ数 × 正解率)
- Runner 間で同じ問題に対する `raw_response_keys` の順序差検出 (warning フラグ集計)
- Model 間の正解率交互作用 (3 model × 4 条件のヒートマップ)
- **ランダム化監査**: `run_order_idx` を 5 等分 (quintile) し、各 quintile × `condition_id` で
  chi-square 検定 (`scipy.stats.chi2_contingency`)。p > 0.05 (= 一様性が棄却されない) を確認。
  Model × `run_order_idx` についても同様に実施し、実行順偏りの不在を担保
- `model_string` (API レスポンスの model フィールド) の stage 内 / stage 間での一致確認。
  不一致があった場合は該当時刻範囲を `summary.md` に列挙

## 4. 出力

`sor-lab report <run_id>` が以下を `results/runs/<run_id>/summary.md` に出力:

1. メタ情報 (stage, 3 model の deployment/api_version/model_string/reasoning_effort, 250 問の question_id 一覧, git commit, EXEC_SEED)
2. **§2.1–2.6 の検定結果テーブル** (条件 × Runner × Model × 指標)
3. ステップ数 × 正解率 折れ線 (Markdown 表 + 任意 PNG)
4. 仮説判定サマリ (H1, H2, H3, H4, H5, H6a, H6b 採択/棄却 + H_QG quality gate 状況)
5. Parse 失敗・key_order_violation の内訳
6. Model 間正解率ヒートマップ
7. ランダム化監査結果
8. 探索的所見 (自由記述用テンプレ)

## 5. 事前登録の取り扱い

- 本書の §2 の検定設計はデータ収集前に確定。
- §3 の探索的分析は事後追加可能だが、§2 の主結論には影響させない。
- 検定の追加・変更は git log と `Revision History` に記録。

## Revision History

- 2026-05-13: 初版。
- 2026-05-13: Phase B/C 構成と軸5 (Model) / 軸6 (Language) 追加に伴い、H1 を Runner × Model 二重 (Bonferroni α'=0.0083) に再校正、H6/H7/H8 を追加。データ単位に `phase` / `model_key` / `lang_sub` を含めた。
- 2026-05-13: 統計レビュー対応の最終改訂。Model 軸を `gpt-4.1` / `gpt-4.1-mini` / `gpt-5` の 3 個に置換。n=50 → n=250。H1 を primary cell (`gpt-4.1 × openai_sdk × det`) 単一検定 + secondary descriptive (8 cell) に再校正。H2/H4 を `proportions_ztost` + Newcombe paired CI による形式 TOST に置き換え (ε はそれぞれ ±3pt / ±5pt)。H3 は Spearman を廃止し per-point Newcombe CI + Cochran-Armitage に変更。H6 を within-gen (6a) + cross-architecture META (6b) に再定義。H7/H8 削除 (言語軸 future work)。`key_order_violation_rate > 1%` で H1 除外する H_QG quality gate を新設。`run_order_idx` を用いたランダム化監査を §3 に追加。データ単位から `phase` / `lang_sub` を削除し `run_order_idx` を追加。
