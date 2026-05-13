"""ペア比較用の信頼区間と TOST ヘルパー。

事前登録に従い、H1/H2/H4 で使う paired difference CI は Newcombe (1998)
method 10 (score-based) を採用する。Wilson CI は単一比率専用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sor_lab.evaluation import wilson_ci


@dataclass(frozen=True)
class PairedDiffCI:
    """対応のある 2 比率の差 (p_A - p_B) の信頼区間。単位は確率 (0..1)。"""

    diff: float
    lower: float
    upper: float


def newcombe_paired_diff_ci(
    e: int, f: int, g: int, h: int, z: float = 1.96
) -> PairedDiffCI:
    """Newcombe (1998) method 10 による paired binomial proportion difference の CI。

    引数は 2x2 contingency table のセル:
        e = A=1, B=1 (両方 success)
        f = A=1, B=0 (A のみ success)
        g = A=0, B=1 (B のみ success)
        h = A=0, B=0 (両方 failure)

    戻り値 `diff = p_A - p_B = (e + f)/n - (e + g)/n = (f - g)/n`。

    n=0 では (0, -1, 1) を返す。各境界は [-1, 1] にクリップする。
    """
    n = e + f + g + h
    if n <= 0:
        return PairedDiffCI(diff=0.0, lower=-1.0, upper=1.0)

    p_a = (e + f) / n
    p_b = (e + g) / n
    diff = p_a - p_b

    wa = wilson_ci(e + f, n, z=z)
    wb = wilson_ci(e + g, n, z=z)

    l1 = p_a - wa.lower
    u1 = wa.upper - p_a
    l2 = p_b - wb.lower
    u2 = wb.upper - p_b

    # 相関補正 phi。分母が 0 のときは 0 として扱う。
    var_a = p_a * (1.0 - p_a)
    var_b = p_b * (1.0 - p_b)
    if var_a > 0 and var_b > 0 and n > 1:
        # 標本共分散: (e*h - f*g) / n^2 (paired indicator の共分散)
        cov = (e * h - f * g) / (n * n)
        phi = cov / math.sqrt(var_a * var_b)
        # 桁あふれを防ぐためにクリップ
        phi = max(-1.0, min(1.0, phi))
    else:
        phi = 0.0

    inner_lower = l1 * l1 - 2.0 * phi * l1 * u2 + u2 * u2
    inner_upper = u1 * u1 - 2.0 * phi * u1 * l2 + l2 * l2
    delta_lower = math.sqrt(max(0.0, inner_lower))
    delta_upper = math.sqrt(max(0.0, inner_upper))

    lower = max(-1.0, diff - delta_lower)
    upper = min(1.0, diff + delta_upper)
    return PairedDiffCI(diff=diff, lower=lower, upper=upper)


def tost_two_proportions(
    succ_a: int,
    n_a: int,
    succ_b: int,
    n_b: int,
    *,
    epsilon: float,
    alpha: float = 0.05,
) -> tuple[float, float, bool]:
    """独立 2 標本の差に対する TOST (Two One-Sided Tests)。

    `proportions_ztost` の薄いラッパ。返り値は `(p_lower, p_upper, equivalent)`。
    両側 p < alpha かつ点推定が ±epsilon 内に収まれば equivalent=True。

    paired データでは使わない (paired には Newcombe CI を使い、CI が
    [-epsilon, epsilon] に内包されるかで判定する)。
    """
    from statsmodels.stats.proportion import proportions_ztost

    counts = [succ_a, succ_b]
    nobs = [n_a, n_b]
    pval, _, _ = proportions_ztost(counts, nobs, low=-epsilon, upp=epsilon)
    p_low = p_up = float(pval)
    diff = (succ_a / n_a if n_a else 0.0) - (succ_b / n_b if n_b else 0.0)
    equivalent = float(pval) < alpha and abs(diff) < epsilon
    return p_low, p_up, equivalent


__all__ = [
    "PairedDiffCI",
    "newcombe_paired_diff_ci",
    "tost_two_proportions",
]
