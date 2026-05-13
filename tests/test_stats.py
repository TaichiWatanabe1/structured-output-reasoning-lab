"""Newcombe paired CI と TOST helper の数値検証。"""

from __future__ import annotations

import pytest

from sor_lab.stats import PairedDiffCI, newcombe_paired_diff_ci


def test_newcombe_returns_paired_diff_ci_type() -> None:
    ci = newcombe_paired_diff_ci(e=10, f=5, g=3, h=2)
    assert isinstance(ci, PairedDiffCI)


def test_newcombe_diff_matches_proportion_diff() -> None:
    # n=20: p_A=15/20=0.75, p_B=13/20=0.65, diff=0.10
    ci = newcombe_paired_diff_ci(e=10, f=5, g=3, h=2)
    assert ci.diff == pytest.approx(0.10, abs=1e-9)


def test_newcombe_ci_includes_diff() -> None:
    ci = newcombe_paired_diff_ci(e=10, f=5, g=3, h=2)
    assert ci.lower <= ci.diff <= ci.upper


def test_newcombe_ci_within_minus_one_to_one() -> None:
    ci = newcombe_paired_diff_ci(e=50, f=0, g=0, h=0)
    # 全 e: p_A=1, p_B=1, diff=0
    assert ci.diff == pytest.approx(0.0)
    assert -1.0 <= ci.lower <= ci.upper <= 1.0


def test_newcombe_zero_n_safe() -> None:
    ci = newcombe_paired_diff_ci(e=0, f=0, g=0, h=0)
    assert ci.diff == 0.0
    assert ci.lower == -1.0
    assert ci.upper == 1.0


def test_newcombe_large_effect_ci_excludes_zero() -> None:
    """n=250, b=10, c=40 (RF 強く有意) のような場合に CI が 0 を含まない方向にあるか。"""
    # A=AF, B=RF と読み替えると、ここでは A=RF, B=AF の差を見たい
    # 関数規約: e=A&B 両方、f=A only、g=B only、h=両方失敗
    # A=RF が +30/250 強いとする: e=200, f=40 (RF only), g=10 (AF only), h=0
    ci = newcombe_paired_diff_ci(e=200, f=40, g=10, h=0)
    # diff = p_A - p_B = (200+40)/250 - (200+10)/250 = 240/250 - 210/250 = 0.12
    assert ci.diff == pytest.approx(0.12, abs=1e-9)
    # CI 下限が 0 より大きい (有意) はず
    assert ci.lower > 0.0


def test_newcombe_symmetric_around_zero_when_no_effect() -> None:
    """f=g かつ e=h のとき、diff ≈ 0、CI は ~0 を中心に対称になる。"""
    ci = newcombe_paired_diff_ci(e=50, f=10, g=10, h=30)
    assert ci.diff == pytest.approx(0.0)
    # 対称性: |lower| ≈ upper
    assert abs(abs(ci.lower) - ci.upper) < 0.05
