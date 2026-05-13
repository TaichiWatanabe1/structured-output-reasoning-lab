"""環境変数 (`.env`) ロード。`pydantic-settings` で型付け済みの設定にする。

軸5 (Model) として `gpt-4.1` / `gpt-4.1-mini` / `gpt-5` (`reasoning_effort=minimal`)
の 3 デプロイを扱う。実行順ランダム化のシード `EXEC_SEED` も同所で管理する。
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# モデルキー (JSONL の `model_key` カラムにそのまま入る値)
MODEL_KEY_GPT41 = "gpt-4.1"
MODEL_KEY_GPT41_MINI = "gpt-4.1-mini"
MODEL_KEY_GPT5 = "gpt-5"

MODEL_KEYS: tuple[str, ...] = (MODEL_KEY_GPT41, MODEL_KEY_GPT41_MINI, MODEL_KEY_GPT5)


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

    deployment_gpt41: str = Field(default="", alias="AZURE_OPENAI_DEPLOYMENT_NAME_GPT41")
    deployment_gpt41_mini: str = Field(
        default="", alias="AZURE_OPENAI_DEPLOYMENT_NAME_GPT41_MINI"
    )
    deployment_gpt5: str = Field(default="", alias="AZURE_OPENAI_DEPLOYMENT_NAME_GPT5")

    reasoning_effort_gpt5: str = Field(
        default="minimal", alias="AZURE_OPENAI_REASONING_EFFORT"
    )

    exec_seed: int = Field(default=42, alias="EXEC_SEED")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    def models(self) -> list[ModelSpec]:
        """設定済みの 3 model spec を返す。未設定モデルは除外する。"""
        out: list[ModelSpec] = []
        if self.deployment_gpt41:
            out.append(ModelSpec(MODEL_KEY_GPT41, self.deployment_gpt41, None))
        if self.deployment_gpt41_mini:
            out.append(ModelSpec(MODEL_KEY_GPT41_MINI, self.deployment_gpt41_mini, None))
        if self.deployment_gpt5:
            out.append(
                ModelSpec(
                    MODEL_KEY_GPT5,
                    self.deployment_gpt5,
                    self.reasoning_effort_gpt5 or "minimal",
                )
            )
        return out

    def is_complete(self) -> bool:
        """API キー・エンドポイントと、少なくとも 1 model 分のデプロイが揃っているか。"""
        return bool(self.api_key and self.endpoint and self.models())

    def missing_fields(self) -> list[str]:
        """未設定の項目名を返す。エラーメッセージ生成用。"""
        missing: list[str] = []
        if not self.api_key:
            missing.append("AZURE_OPENAI_API_KEY")
        if not self.endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not (self.deployment_gpt41 or self.deployment_gpt41_mini or self.deployment_gpt5):
            missing.append(
                "AZURE_OPENAI_DEPLOYMENT_NAME_GPT41 / _GPT41_MINI / _GPT5 (少なくとも 1 つ)"
            )
        return missing


def load_settings() -> AzureOpenAISettings:
    return AzureOpenAISettings()


__all__ = [
    "MODEL_KEY_GPT41",
    "MODEL_KEY_GPT41_MINI",
    "MODEL_KEY_GPT5",
    "MODEL_KEYS",
    "ModelSpec",
    "AzureOpenAISettings",
    "load_settings",
]
