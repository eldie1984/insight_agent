"""Configuration for the Sales Assistant Agent."""

import os
import logging
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment variables."""

    # Core settings
    environment: str = os.getenv("ENVIRONMENT", "development")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # OpenRouter (provides access to Claude, GPT, and other models)
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    # Using GPT-3.5-turbo (cheap, reliable tool calling on OpenRouter)
    # Alternative: "anthropic/claude-3.5-sonnet" (paid but best quality)
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Google Cloud
    gcp_project_id: str = os.getenv("GCP_PROJECT_ID", "")
    gcp_region: str = os.getenv("GCP_REGION", "us-central1")

    # BigQuery
    bigquery_dataset: str = os.getenv("BIGQUERY_DATASET", "sales_data")
    bigquery_table: str = os.getenv("BIGQUERY_TABLE", "sales_fact_table")

    # Forecast model service
    forecast_model_endpoint: str = os.getenv(
        "FORECAST_MODEL_ENDPOINT",
        "http://localhost:8080/predict_array",
    )

    # LangSmith Observability
    langsmith_tracing: bool = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "sales-assistant-agent")

    # Google Cloud Authentication
    google_application_credentials: str = os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS", ""
    )

    class Config:
        env_file = ".env"
        case_sensitive = False

    def setup_logging(self) -> None:
        """Configure logging based on settings."""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def setup_langsmith(self) -> None:
        """Configure LangSmith observability."""
        if self.langsmith_tracing:
            if not self.langsmith_api_key:
                raise ValueError(
                    "LANGSMITH_API_KEY must be set when LANGSMITH_TRACING=true"
                )
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
            os.environ["LANGSMITH_PROJECT"] = self.langsmith_project

            logging.info(
                f"LangSmith tracing enabled for project: {self.langsmith_project}"
            )


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


# Initialize settings on module import
settings = get_settings()
