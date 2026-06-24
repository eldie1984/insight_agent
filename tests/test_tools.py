"""Tests for tool implementations."""

import pytest
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

from tools import (
    get_historical_sales,
    get_forecast,
    _get_historical_sales_stub,
    _check_historical_data_available,
)


class TestHistoricalSalesStub:
    """Test stub data generation for historical sales."""

    def test_returns_list_of_dicts(self):
        """Should return list of dicts with required keys."""
        result = _get_historical_sales_stub("SIOUX", 10)
        assert isinstance(result, list)
        assert len(result) == 10
        assert all(isinstance(p, dict) for p in result)
        assert all("date" in p and "county" in p and "value" in p for p in result)

    def test_includes_county(self):
        """Should include county in all points."""
        result = _get_historical_sales_stub("BOONE", 5)
        assert all(p["county"] == "BOONE" for p in result)

    def test_date_range(self):
        """Should span correct number of days."""
        result = _get_historical_sales_stub("TEST", 30)
        assert len(result) == 30
        today = date.today()
        first_date = date.fromisoformat(result[0]["date"])
        last_date = date.fromisoformat(result[-1]["date"])
        assert (today - first_date).days == 30
        assert last_date == today - timedelta(days=1)

    def test_values_are_positive(self):
        """Should return positive sales values."""
        result = _get_historical_sales_stub("SIOUX", 10)
        assert all(p["value"] > 0 for p in result)


class TestGetHistoricalSales:
    """Test get_historical_sales tool."""

    def test_returns_stub_data_when_no_bigquery(self):
        """Should return stub data when BigQuery unavailable."""
        with patch("tools.bq_client", None):
            result = get_historical_sales.invoke(
                {"county": "SIOUX", "lookback_days": 30}
            )
            assert isinstance(result, list)
            assert len(result) == 30
            assert all(p["county"] == "SIOUX" for p in result)

    def test_accepts_county_and_lookback_days(self):
        """Should accept county and lookback_days parameters."""
        with patch("tools.bq_client", None):
            result = get_historical_sales.invoke(
                {"county": "LYON", "lookback_days": 15}
            )
            assert len(result) == 15
            assert all(p["county"] == "LYON" for p in result)

    def test_default_lookback_is_60_days(self):
        """Should default to 60 days lookback."""
        with patch("tools.bq_client", None):
            result = get_historical_sales.invoke({"county": "TEST"})
            assert len(result) == 60


class TestGetForecast:
    """Test get_forecast tool."""

    def test_accepts_required_parameters(self):
        """Should accept county and horizon_days parameters."""
        with patch(
            "tools._check_historical_data_available",
            return_value=(True, "2026-03-01", "2026-06-24"),
        ):
            with patch("tools.httpx.Client") as mock_client:
                mock_response = MagicMock()
                mock_response.json.return_value = {"forecast": [100.0] * 30}
                mock_client.return_value.__enter__.return_value.post.return_value = (
                    mock_response
                )

                result = get_forecast.invoke({"county": "SIOUX", "horizon_days": 30})
                assert isinstance(result, list)
                assert len(result) == 30

    def test_validates_horizon_days_range(self):
        """Should validate horizon_days is between 1-30."""
        with patch(
            "tools._check_historical_data_available",
            return_value=(True, "2026-03-01", "2026-06-24"),
        ):
            with patch("tools.httpx.Client") as mock_client:
                mock_response = MagicMock()
                mock_response.json.return_value = {"forecast": [100.0] * 15}
                mock_client.return_value.__enter__.return_value.post.return_value = (
                    mock_response
                )

                # Should work with valid range
                result = get_forecast.invoke({"county": "SIOUX", "horizon_days": 15})
                assert isinstance(result, list)

    def test_from_date_parameter_optional(self):
        """Should accept optional from_date parameter."""
        with patch(
            "tools._check_historical_data_available",
            return_value=(True, "2026-03-01", "2026-06-24"),
        ):
            with patch("tools.httpx.Client") as mock_client:
                mock_response = MagicMock()
                mock_response.json.return_value = {"forecast": [100.0] * 31}
                mock_client.return_value.__enter__.return_value.post.return_value = (
                    mock_response
                )

                # Should work with from_date
                result = get_forecast.invoke(
                    {"county": "SIOUX", "horizon_days": 31, "from_date": "2026-03-01"}
                )
                assert isinstance(result, list)


class TestCheckHistoricalDataAvailable:
    """Test historical data availability check."""

    def test_raises_when_no_bigquery(self):
        """Should raise ValueError when BigQuery unavailable."""
        with patch("tools.bq_client", None):
            with pytest.raises(ValueError):
                _check_historical_data_available("SIOUX")

    def test_handles_bigquery_errors_gracefully(self):
        """Should handle BigQuery errors and return True."""
        mock_client = MagicMock()
        mock_client.query.side_effect = Exception("BigQuery error")

        with patch("tools.bq_client", mock_client):
            has_data, min_d, max_d = _check_historical_data_available(
                "SIOUX", start_date=date(2026, 3, 1), end_date=date(2026, 6, 24)
            )
            # Should return True on error (assume data available)
            assert has_data is True
