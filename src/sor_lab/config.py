"""環境変数 (`.env`) ロード。`pydantic-settings` で型付け済みの設定にする。"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AzureOpenAISettings(BaseSettings):
    """Azure OpenAI への接続情報。`.env` から自動ロード。"""

    api_key: str = Field(default="", alias="AZURE_OPENAI_API_KEY")
    endpoint: str = Field(default="", alias="AZURE_OPENAI_ENDPOINT")
    api_version: str = Field(default="2024-10-21", alias="AZURE_OPENAI_API_VERSION")
    deployment_name: str = Field(default="", alias="AZURE_OPENAI_DEPLOYMENT_NAME")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    def is_complete(self) -> bool:
        return bool(self.api_key and self.endpoint and self.deployment_name)


def load_settings() -> AzureOpenAISettings:
    return AzureOpenAISettings()  # type: ignore[call-arg]
