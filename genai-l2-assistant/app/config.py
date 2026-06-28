"""Application configuration using pydantic-settings.

All settings are loaded from environment variables with sensible defaults
for local development. In production, configure via .env file or
container environment variables.
"""

from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class VectorStoreProvider(str, Enum):
    """Supported vector store backends."""
    PINECONE = "pinecone"
    PGVECTOR = "pgvector"


class AppEnvironment(str, Enum):
    """Application deployment environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogFormat(str, Enum):
    """Log output format."""
    HUMAN = "human"
    JSON = "json"


class LLMSettings(BaseSettings):
    """LLM provider configuration.

    Supports OpenAI (direct or Azure) and Anthropic Claude.
    When using Azure OpenAI, set azure_endpoint to enable Azure mode.
    """
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        alias="LLM_PROVIDER",
        description="LLM provider: openai or anthropic",
    )
    model_name: str = Field(
        default="gpt-4o",
        alias="LLM_MODEL_NAME",
        description="Model identifier (e.g., gpt-4o, claude-3-5-sonnet-20241022)",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        alias="OPENAI_API_KEY",
        description="OpenAI direct API key",
    )
    azure_openai_api_key: Optional[str] = Field(
        default=None,
        alias="AZURE_OPENAI_API_KEY",
        description="Azure OpenAI API key",
    )
    azure_endpoint: Optional[str] = Field(
        default=None,
        alias="AZURE_OPENAI_ENDPOINT",
        description="Azure OpenAI endpoint URL",
    )
    azure_api_version: str = Field(
        default="2024-02-01",
        alias="AZURE_OPENAI_API_VERSION",
        description="Azure OpenAI API version",
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        alias="ANTHROPIC_API_KEY",
        description="Anthropic API key",
    )

    @property
    def is_azure(self) -> bool:
        """Check if Azure OpenAI should be used."""
        return self.azure_endpoint is not None and self.azure_openai_api_key is not None


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", extra="ignore")

    model: str = Field(
        default="text-embedding-3-large",
        alias="EMBEDDING_MODEL",
        description="Embedding model name",
    )
    dimensions: int = Field(
        default=3072,
        alias="EMBEDDING_DIMENSIONS",
        description="Embedding vector dimensions",
    )


class VectorStoreSettings(BaseSettings):
    """Vector store configuration for Pinecone or pgvector."""
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    provider: VectorStoreProvider = Field(
        default=VectorStoreProvider.PINECONE,
        alias="VECTOR_STORE_PROVIDER",
        description="Vector store provider: pinecone or pgvector",
    )
    pinecone_api_key: Optional[str] = Field(
        default=None,
        alias="PINECONE_API_KEY",
        description="Pinecone API key",
    )
    pinecone_index_name: str = Field(
        default="l2-assistant-index",
        alias="PINECONE_INDEX_NAME",
        description="Pinecone index name",
    )
    pinecone_environment: str = Field(
        default="us-east-1-aws",
        alias="PINECONE_ENVIRONMENT",
        description="Pinecone environment/region",
    )


class ServiceNowSettings(BaseSettings):
    """ServiceNow instance connection settings."""
    model_config = SettingsConfigDict(env_prefix="SNOW_", extra="ignore")

    instance_url: str = Field(
        default="https://dev-instance.service-now.com",
        alias="SNOW_INSTANCE_URL",
        description="ServiceNow instance base URL",
    )
    username: Optional[str] = Field(
        default=None,
        alias="SNOW_USERNAME",
        description="ServiceNow service account username",
    )
    password: Optional[str] = Field(
        default=None,
        alias="SNOW_PASSWORD",
        description="ServiceNow service account password",
    )
    client_id: Optional[str] = Field(
        default=None,
        alias="SNOW_CLIENT_ID",
        description="OAuth client ID",
    )
    client_secret: Optional[str] = Field(
        default=None,
        alias="SNOW_CLIENT_SECRET",
        description="OAuth client secret",
    )
    webhook_secret: Optional[str] = Field(
        default=None,
        alias="SNOW_WEBHOOK_SECRET",
        description="HMAC secret for webhook validation",
    )

    @property
    def use_oauth(self) -> bool:
        """Check if OAuth credentials are configured."""
        return self.client_id is not None and self.client_secret is not None


class DatabaseSettings(BaseSettings):
    """Database connection settings for PostgreSQL and Redis."""
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    postgres_url: str = Field(
        default="postgresql+asyncpg://l2assistant:l2assistant@localhost:5432/l2assistant",
        alias="DATABASE_URL",
        description="PostgreSQL async connection string",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        alias="REDIS_URL",
        description="Redis connection string",
    )

    @property
    def sync_postgres_url(self) -> str:
        """Return synchronous Postgres URL for Alembic migrations."""
        return self.postgres_url.replace("+asyncpg", "")


class ObservabilitySettings(BaseSettings):
    """Observability and tracing configuration."""
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    langsmith_api_key: Optional[str] = Field(
        default=None,
        alias="LANGSMITH_API_KEY",
        description="LangSmith API key for LLM tracing",
    )
    langsmith_project: str = Field(
        default="l2-assistant-dev",
        alias="LANGSMITH_PROJECT",
        description="LangSmith project name",
    )
    prometheus_enabled: bool = Field(
        default=True,
        alias="PROMETHEUS_ENABLED",
        description="Enable Prometheus metrics endpoint",
    )


class AppSettings(BaseSettings):
    """Root application settings that composes all sub-settings.

    Usage:
        settings = get_settings()
        settings.llm.provider  # LLMProvider.OPENAI
        settings.database.postgres_url  # connection string
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_env: AppEnvironment = Field(
        default=AppEnvironment.DEVELOPMENT,
        alias="APP_ENV",
        description="Deployment environment",
    )
    log_format: LogFormat = Field(
        default=LogFormat.HUMAN,
        alias="LOG_FORMAT",
        description="Log output format",
    )
    secret_key: str = Field(
        default="change-this-to-a-random-secret-key",
        alias="SECRET_KEY",
        description="Application secret key",
    )
    allowed_origins: str = Field(
        default="http://localhost:3000",
        alias="ALLOWED_ORIGINS",
        description="Comma-separated CORS allowed origins",
    )

    # Composed sub-settings
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    servicenow: ServiceNowSettings = Field(default_factory=ServiceNowSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @property
    def cors_origins(self) -> list[str]:
        """Parse allowed origins into a list."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.app_env == AppEnvironment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.app_env == AppEnvironment.DEVELOPMENT


@lru_cache()
def get_settings() -> AppSettings:
    """Get cached application settings singleton.

    Returns:
        AppSettings: The application settings loaded from environment.
    """
    return AppSettings()
