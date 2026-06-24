"""Shared test fixtures and configuration."""

import pytest
import sys
from pathlib import Path

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_settings():
    """Mock application settings."""
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock.openrouter_api_key = "test-key"
    mock.openrouter_model = "openai/gpt-3.5-turbo"
    mock.openrouter_base_url = "https://openrouter.ai/api/v1"
    mock.gcp_project_id = "test-project"
    mock.bigquery_dataset = "test_dataset"
    mock.bigquery_table = "test_table"
    mock.forecast_model_endpoint = "http://localhost:8080/predict_array"
    mock.langsmith_tracing = False
    mock.langsmith_api_key = ""

    return mock


@pytest.fixture
def sample_historical_data():
    """Sample historical sales data."""
    return [
        {"date": "2026-02-24", "county": "SIOUX", "value": 25000.0},
        {"date": "2026-02-25", "county": "SIOUX", "value": 26000.0},
        {"date": "2026-02-26", "county": "SIOUX", "value": 24000.0},
    ]


@pytest.fixture
def sample_forecast_data():
    """Sample forecast data."""
    return [
        {"date": "2026-06-25", "value": 25000.0},
        {"date": "2026-06-26", "value": 26000.0},
        {"date": "2026-06-27", "value": 24000.0},
    ]


@pytest.fixture
def client():
    """FastAPI test client."""
    from fastapi.testclient import TestClient
    from main import app

    return TestClient(app)
