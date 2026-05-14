"""5 条件分のシステム / ユーザープロンプトを正本として保持する (日本語固定)。

実験条件のあいだで「reasoning を促す表現」を厳密に揃えるためのモジュール。
本実験では入出力言語を日本語に固定する (本番システムが日本語向けのため)。
変更時は docs/experiment_design.md §7 のリビジョンも更新する。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    """system / user テンプレ。`{question}` を `format` で差し込んで使う。"""

    system: str
    user: str

    def render(self, question: str) -> tuple[str, str]:
        return self.system, self.user.format(question=question)


_USER_TEMPLATE = "{question}"

# Structured Output 3 条件で共有するシステムプロンプト (順序のみが独立変数)
_SO_SYSTEM = "あなたは算数の教師です。スキーマに一致する JSON を返してください。"

PROMPTS: dict[str, PromptTemplate] = {
    "plain_direct": PromptTemplate(
        system=(
            "あなたは算数の教師です。最終的な整数の答えだけを出力してください。"
            "他の文字は一切含めないでください。"
        ),
        user=_USER_TEMPLATE,
    ),
    "plain_cot": PromptTemplate(
        system=(
            "あなたは算数の教師です。順を追って考え、最後の行に整数の答えを "
            "「答え: 」の後に記載してください。"
        ),
        user=_USER_TEMPLATE,
    ),
    "so_answer_first": PromptTemplate(
        system=_SO_SYSTEM,
        user=_USER_TEMPLATE,
    ),
    "so_reasoning_first": PromptTemplate(
        system=_SO_SYSTEM,
        user=_USER_TEMPLATE,
    ),
    "so_step_constrained": PromptTemplate(
        system=_SO_SYSTEM,
        user=_USER_TEMPLATE,
    ),
    # b0 文献比較校正条件 (Shi et al., 2022 / Kojima et al., 2022 相当)
    # 主仮説の検定対象ではなく、公表 MGSM accuracy との比較用。
    "lit_mgsm_direct_ja": PromptTemplate(
        system="問題を解いてください。整数の答えだけを出力してください。",
        user=_USER_TEMPLATE,
    ),
    "lit_mgsm_native_cot_ja": PromptTemplate(
        system=(
            "問題をステップごとに解き、最後に「答え: 数字」の形式で"
            "整数の答えを書いてください。"
        ),
        user=_USER_TEMPLATE,
    ),
}


def get_prompt(condition_id: str) -> PromptTemplate:
    """`condition_id` の base 部分 (`:` の前) を見て対応するテンプレを返す。

    `so_step_constrained:n_steps=5:answer_first=False` のような派生 ID にも対応。
    """
    base = condition_id.split(":", 1)[0]
    try:
        return PROMPTS[base]
    except KeyError as exc:
        raise KeyError(f"unknown condition_id base: {base!r}") from exc
