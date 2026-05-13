"""Runner: LLM プロバイダ別の呼び出し実装。

3 種類とも `Runner` Protocol を実装し、同じ入力に対して `RunnerResult` を返す。
"""

from sor_lab.runners.base import Runner, RunnerResult

__all__ = ["Runner", "RunnerResult"]
