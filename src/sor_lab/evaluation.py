"""答え抽出 + 正誤判定 + ステップ数カウント + 統計指標。"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


# GSM8K の reference answer 末尾の "#### 18" 形式から整数を抽出する。
_GOLD_ANSWER_RE = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")

# Plain 系応答から末尾整数を拾う。"答え: 42" / "Answer: 42." や末尾改行も許容。
# 1) "答え: <num>" または "Answer: <num>" を最優先 (plain_cot は 答え:、fallback で Answer: も受理)
# 2) 末尾の整数を fallback
_ANSWER_PREFIX_RE = re.compile(
    r"(?:答え|Answer)\s*[:：]\s*\$?(-?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
_TRAILING_NUM_RE = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)\D*\Z")


def parse_gold_answer(reference: str) -> int:
    """GSM8K reference answer から正解整数を抽出。"""
    m = _GOLD_ANSWER_RE.search(reference)
    if not m:
        raise ValueError(f"could not find #### gold answer in: {reference!r}")
    return _to_int(m.group(1))


def parse_plain_answer(text: str) -> int | None:
    """Plain 系応答からモデルの整数解答を抽出。失敗時は None。"""
    if not text:
        return None

    # 1) "Answer: N" を優先
    m = _ANSWER_PREFIX_RE.search(text)
    if m:
        try:
            return _to_int(m.group(1))
        except ValueError:
            pass

    # 2) 末尾の整数 fallback
    m = _TRAILING_NUM_RE.search(text.strip())
    if m:
        try:
            return _to_int(m.group(1))
        except ValueError:
            return None

    return None


def _to_int(raw: str) -> int:
    """`"1,200"` や `"3.0"` を int に。小数点で非ゼロなら ValueError。"""
    cleaned = raw.replace(",", "")
    if "." in cleaned:
        f = float(cleaned)
        i = int(round(f))
        if not math.isclose(f, i, abs_tol=1e-9):
            raise ValueError(f"non-integer float: {raw!r}")
        return i
    return int(cleaned)


def count_reasoning_steps(reasoning: str | list[str] | None) -> int:
    """ステップ数をカウント。

    - `list[str]` ならその長さ
    - `str` なら改行 or 番号付き行 (`1. ...`) を区切りとしてカウント
    - None は 0
    """
    if reasoning is None:
        return 0
    if isinstance(reasoning, list):
        return len(reasoning)

    # 改行ベース。空行は無視
    lines = [line for line in re.split(r"[\r\n]+", reasoning) if line.strip()]
    if not lines:
        return 0

    # 番号付き行 (`1.` `1)` `(1)`) が複数あればそれを優先
    numbered = [line for line in lines if re.match(r"^\s*\(?\d+[.)]", line)]
    if len(numbered) >= 2:
        return len(numbered)
    return len(lines)


# ---- aggregation -------------------------------------------------------------


@dataclass(frozen=True)
class WilsonCI:
    """Wilson スコア法による二項比例の信頼区間。"""

    lower: float
    upper: float


def wilson_ci(successes: int, n: int, z: float = 1.96) -> WilsonCI:
    """`z=1.96` で 95% Wilson CI。`n=0` のときは `(0, 1)` を返す。"""
    if n <= 0:
        return WilsonCI(0.0, 1.0)
    p = successes / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n))) / denom
    return WilsonCI(max(0.0, center - half), min(1.0, center + half))


def majority_vote(answers: list[int | None]) -> int | None:
    """Self-consistency 多数決。`None` は除外。全部 None なら None。"""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    counts = Counter(valid)
    # 最頻値が複数あれば最小の値を返す (決定的)
    top_count = max(counts.values())
    candidates = [a for a, c in counts.items() if c == top_count]
    return min(candidates)


__all__ = [
    "parse_gold_answer",
    "parse_plain_answer",
    "count_reasoning_steps",
    "wilson_ci",
    "majority_vote",
    "WilsonCI",
]
