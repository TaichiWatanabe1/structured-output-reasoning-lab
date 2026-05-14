# Appendix — Prompt & Schema Canonical Text

本実験で固定されるすべての **prompt 文** ・ **Pydantic schema** ・ **decoding params** を本ファイルに集約する。事前登録 (NeurIPS / ICLR / Responsible NLP checklist) の透明性要件に沿って、これらは実験条件そのものとして扱う。

- 正本コードは [`src/sor_lab/prompts.py`](../src/sor_lab/prompts.py) と [`src/sor_lab/schemas.py`](../src/sor_lab/schemas.py)。
- 本ファイルとコードの diff は [`tests/test_prompts.py`](../tests/test_prompts.py) の SHA-256 digest assert で監視する。
- 文言を変更する場合は **データ収集前に** 本ファイル・コード・テスト digest を同 PR で更新する。

---

## 1. System / User Prompts (5 conditions, 日本語固定)

入出力言語は日本語固定 (本番システムが日本語向けのため)。`{question}` は MGSM-ja の問題文 (日本語) に置換される。

### 1.1 `plain_direct` — Plain text, no CoT

```text
system: あなたは算数の教師です。最終的な整数の答えだけを出力してください。他の文字は一切含めないでください。
user: {question}
```

### 1.2 `plain_cot` — Plain text, CoT

```text
system: あなたは算数の教師です。順を追って考え、最後の行に整数の答えを「答え: 」の後に記載してください。
user: {question}
```

answer marker は日本語ネイティブ `答え:` (2026-05-14 変更; 旧: `Answer:`)。評価は `_ANSWER_PREFIX_RE` で `答え:` / `Answer:` 両方を受理。

### 1.3 Structured Output 系 (`so_*`) で共有する system prompt

**`so_answer_first` / `so_reasoning_first` / `so_step_constrained` の 3 条件で system prompt は完全に同一文字列**。差分は **Pydantic schema の field 順序のみ** (`so_step_constrained` のみ schema 形状も異なる)。これは順序効果以外の交絡を排するための統制。

```text
system: あなたは算数の教師です。スキーマに一致する JSON を返してください。
user: {question}
```

### 1.4 `so_step_constrained` — Structured Output, step-constrained schema design

`so_answer_first` / `so_reasoning_first` と **system / user prompt は同一**。差分は schema の `reasoning_steps: list[str]` フィールドと `max_length` 制約のみ。

### 1.5 `lit_mgsm_direct_ja` — 文献比較校正 (b0、主仮説対象外)

```text
system: 問題を解いてください。整数の答えだけを出力してください。
user: {question}
```

参照: Shi et al., 2022 (MGSM) direct format 相当。公表 MGSM 精度との比較用。

### 1.6 `lit_mgsm_native_cot_ja` — 文献比較校正 (b0、主仮説対象外)

```text
system: 問題をステップごとに解き、最後に「答え: 数字」の形式で整数の答えを書いてください。
user: {question}
```

参照: Shi et al., 2022 (MGSM) / Kojima et al., 2022 (zero-shot CoT) 相当。answer marker は `答え:` (日本語ネイティブ)。

---

## 2. Pydantic Schemas

`field 名は英語固定` (`reasoning` / `answer` / `reasoning_steps`)、`Field(description=...)` は日本語固定。`ConfigDict(extra="forbid")` で `additionalProperties: false` を強制 (OpenAI strict mode 要件)。

### 2.1 `AnswerFirst` (used by `so_answer_first`)

```python
class AnswerFirst(_StrictModel):
    answer: int = Field(description="最終解答（整数）")
    reasoning: str = Field(description="計算過程を順を追って")
```

### 2.2 `ReasoningFirst` (used by `so_reasoning_first`)

```python
class ReasoningFirst(_StrictModel):
    reasoning: str = Field(description="計算過程を順を追って")
    answer: int = Field(description="最終解答（整数）")
```

**`AnswerFirst` と `ReasoningFirst` で:**
- field 名 (`answer` / `reasoning`) は完全一致
- 各 field の `description` 文言も完全一致
- 差分は **declared field の宣言順のみ** (= JSON property の生成順)

これが H1 の独立変数。

### 2.3 `make_step_constrained(n_steps, answer_first)` (used by `so_step_constrained`)

```python
def make_step_constrained(n_steps: int, answer_first: bool) -> type[BaseModel]:
    steps_field = (
        list[str],
        Field(
            description="計算過程のステップ列",
            min_length=1,
            max_length=n_steps,
        ),
    )
    answer_field = (int, Field(description="最終解答（整数）"))
    # answer_first=True → {answer, reasoning_steps}
    # answer_first=False → {reasoning_steps, answer}
```

本実験では `answer_first=False` 固定で `n_steps ∈ {1, 3, 5, 10}` の 4 サブ条件を測る。`answer_first=True` バリエーションは Future Work。

---

## 3. Decoding Parameters

| Parameter | `det` | `sc` | 備考 |
|---|---|---|---|
| `temperature` | `0.0` | `0.7` | reasoning モデル (`gpt-5`) では未指定 (API 非対応) |
| `seed` | `42` (= `dataset_seed`) | **未指定** | `sc` は独立サンプル性のため明示的に seed を渡さない (best-effort 決定性のみ) |
| `n_tries` | `1` | `5` | majority vote は `parsed_answer` の最頻値 (ties は最小値) |
| `response_format` (SO 系) | `{"type": "json_schema", "json_schema": {"name": <cls>, "schema": <strict JSON Schema>, "strict": true}}` | 同左 | OpenAI / Azure 共通。LangChain は `with_structured_output(method="json_schema", strict=True)` |
| `reasoning_effort` | `minimal` (gpt-5 のみ) | `minimal` (gpt-5 のみ) | `gpt-4.1` / `gpt-4.1-mini` には渡さない |

実装上の根拠は [`src/sor_lab/experiment.py`](../src/sor_lab/experiment.py) と [`src/sor_lab/runners/`](../src/sor_lab/runners/) を参照。`call_seed` の分岐は `experiment.py:run_stage` 内の `temp.n_tries == 1` で実施。

---

## 4. Schema Validation (runtime adjudication)

各 Runner は API レスポンスを `schema_cls.model_validate(data)` に通してから採点する。`ValidationError` は `parse_failed=True` として denominator から除外。具体的に弾かれるケース:

- `answer` 欠落 / 型違い (例: `"42"` 文字列、`null`)
- `reasoning` / `reasoning_steps` 欠落 / 型違い
- 余分なキー (`extra="forbid"`)
- `reasoning_steps` の `min_length` / `max_length` 違反 (`so_step_constrained:n_steps=N` で list 長 > N の応答)

`raw_response_keys` は validation の **前** に `json.loads(...).keys()` の挿入順で取得する (= モデルが実際に生成した property 順)。`key_order_violation` は宣言順と異なる場合に立つ。

## 5. Runtime quality gate (H1 主検定)

`docs/experiment_design.md` §8 の規定どおり、SO 両条件 (`so_answer_first` / `so_reasoning_first`) のいずれかで cell 単位の `key_order_violation` 違反率が **> 1%** の場合、その cell は H1 主検定 (McNemar exact + Newcombe paired CI) の対象から **自動除外** され、`summary.md` の H1 表で `excluded_by_key_order_gate=True` / `mcnemar_p=NaN` として明示される。
