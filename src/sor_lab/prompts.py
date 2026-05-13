"""5 条件分のシステム / ユーザープロンプトを正本として保持する。

実験条件のあいだで「reasoning を促す表現」を厳密に揃えるためのモジュール。
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

PROMPTS: dict[str, PromptTemplate] = {
    "plain_direct": PromptTemplate(
        system=(
            "You are a math tutor. Answer with only the final integer, nothing else."
        ),
        user=_USER_TEMPLATE,
    ),
    "plain_cot": PromptTemplate(
        system=(
            "You are a math tutor. Think step by step, then write the final integer "
            "on the last line prefixed by 'Answer: '."
        ),
        user=_USER_TEMPLATE,
    ),
    "so_answer_first": PromptTemplate(
        system="You are a math tutor. Return JSON matching the schema.",
        user=_USER_TEMPLATE,
    ),
    "so_reasoning_first": PromptTemplate(
        system="You are a math tutor. Return JSON matching the schema.",
        user=_USER_TEMPLATE,
    ),
    "so_step_constrained": PromptTemplate(
        system="You are a math tutor. Return JSON matching the schema.",
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
