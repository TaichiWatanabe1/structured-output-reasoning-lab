"""Runner Protocol と共通データ型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RunnerResult:
    """1 コールの結果。後段で JSONL に変換される。"""

    raw_response: str
    """生のレスポンス文字列。Plain 系は本文、Structured 系は JSON 文字列。"""

    raw_response_keys: list[str] | None
    """Structured 出力のキー順序 (`json.loads` で得た dict の挿入順)。
    Plain 系・parse 失敗時は None。"""

    parsed_answer: int | None
    """抽出された整数解答。失敗時 None。"""

    parsed_reasoning: str | list[str] | None
    """抽出された reasoning。Plain 系は本文文字列、Structured は str か list[str]。"""

    parse_failed: bool
    """JSON parse / answer 抽出が失敗したか。"""

    latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None

    model_string: str | None = None
    """API レスポンスの `model` フィールド (例: `gpt-4.1-2025-04-14`)。
    Azure 内部でモデルバージョンが更新されたかの検出に使う。"""


class Runner(Protocol):
    """LLM 呼び出し抽象。実装は `openai_sdk` / `azure_sdk` / `langchain`。"""

    name: str

    def run(
        self,
        condition_id: str,
        question: str,
        temperature: float,
        seed: int | None = None,
    ) -> RunnerResult:
        """1 問を 1 試行回す。"""
        ...
