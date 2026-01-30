"""
Q-Ravens Configuration Module

Handles all configuration settings using pydantic-settings.
Supports multiple LLM providers as specified in the PRD.
"""

from enum import Enum
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    GROQ = "groq"
    AZURE = "azure"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Provider API Keys
    anthropic_api_key: Optional[SecretStr] = Field(default=None)
    openai_api_key: Optional[SecretStr] = Field(default=None)
    google_api_key: Optional[SecretStr] = Field(default=None)
    groq_api_key: Optional[SecretStr] = Field(default=None)
    azure_openai_api_key: Optional[SecretStr] = Field(default=None)
    azure_openai_endpoint: Optional[str] = Field(default=None)

    # Default LLM Settings
    default_llm_provider: LLMProvider = Field(default=LLMProvider.ANTHROPIC)
    default_model: str = Field(default="claude-sonnet-4-20250514")

    # Ollama Settings (Local LLM)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3")

    # Database & Storage
    database_url: Optional[str] = Field(default=None)
    chroma_persist_dir: str = Field(default="./data/chroma")

    # Observability
    langchain_tracing_v2: bool = Field(default=False)
    langchain_api_key: Optional[SecretStr] = Field(default=None)
    langchain_project: str = Field(default="q-ravens")

    # Application Settings
    log_level: str = Field(default="INFO")
    max_browsers: int = Field(default=5)
    default_timeout: int = Field(default=30)

    def get_api_key(self, provider: Optional[LLMProvider] = None) -> Optional[str]:
        """Get the API key for the specified or default provider."""
        provider = provider or self.default_llm_provider

        key_map = {
            LLMProvider.ANTHROPIC: self.anthropic_api_key,
            LLMProvider.OPENAI: self.openai_api_key,
            LLMProvider.GOOGLE: self.google_api_key,
            LLMProvider.GROQ: self.groq_api_key,
            LLMProvider.AZURE: self.azure_openai_api_key,
            LLMProvider.OLLAMA: None,  # Ollama doesn't need an API key
        }

        secret = key_map.get(provider)
        return secret.get_secret_value() if secret else None

    def validate_provider_config(self, provider: Optional[LLMProvider] = None) -> bool:
        """Check if the provider has valid configuration."""
        provider = provider or self.default_llm_provider

        if provider == LLMProvider.OLLAMA:
            return bool(self.ollama_base_url)

        if provider == LLMProvider.AZURE:
            return bool(self.azure_openai_api_key and self.azure_openai_endpoint)

        return bool(self.get_api_key(provider))


# Global settings instance
settings = Settings()
