"""環境変数 (`.env`) ロード。`pydantic-settings` で型付け済みの設定にする。

`.env` は **credentials と環境依存値のみ** を扱う:
- `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_VERSION`
- `EXEC_SEED` (実行順ランダム化シード)

軸5 (Model: key / deployment / reasoning_effort) は `conditions.yaml` の
`models:` セクションで宣言する。
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class ModelSpec:
    """1 つのモデルデプロイメントの解決済み設定。

    `reasoning_effort` は gpt-5 系のみ非 None で、他モデルへは渡さない。
    """

    key: str
    deployment: str
    reasoning_effort: str | None


class AzureOpenAISettings(BaseSettings):
    """Azure OpenAI への接続情報。`.env` から自動ロード。"""

    api_key: str = Field(default="", alias="AZURE_OPENAI_API_KEY")
    endpoint: str = Field(default="", alias="AZURE_OPENAI_ENDPOINT")
    api_version: str = Field(default="2025-04-01-preview", alias="AZURE_OPENAI_API_VERSION")

    exec_seed: int = Field(default=42, alias="EXEC_SEED")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    def is_complete(self) -> bool:
        """API キーとエンドポイントが揃っているか。"""
        return bool(self.api_key and self.endpoint)

    def missing_fields(self) -> list[str]:
        """未設定の credentials 名を返す。エラーメッセージ生成用。"""
        missing: list[str] = []
        if not self.api_key:
            missing.append("AZURE_OPENAI_API_KEY")
        if not self.endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        return missing


def load_settings() -> AzureOpenAISettings:
    return AzureOpenAISettings()


__all__ = [
    "ModelSpec",
    "AzureOpenAISettings",
    "load_settings",
]
