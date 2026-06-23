"""Tool implementations for BigQuery historical sales and model service forecasts."""
import os
import logging
from datetime import date, timedelta
from pydantic import BaseModel, Field
from langchain_core.tools import tool

try:
    from google.cloud import bigquery
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False

import httpx
import time
from config import settings

logger = logging.getLogger(__name__)

# Configuration from environment
GCP_PROJECT_ID = settings.gcp_project_id
BIGQUERY_DATASET = settings.bigquery_dataset
BIGQUERY_TABLE = settings.bigquery_table
MODEL_ENDPOINT = settings.forecast_model_endpoint


# Initialize BigQuery client if credentials available
if BIGQUERY_AVAILABLE and GCP_PROJECT_ID:
    try:
        bq_client = bigquery.Client()
        logger.info(f"BigQuery client initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize BigQuery client: {e}")
        bq_client = None
else:
    bq_client = None
    if not BIGQUERY_AVAILABLE:
        logger.warning("google-cloud-bigquery not installed, using stub data")


class HistoricalSalesInput(BaseModel):
    county: str = Field(..., description="County name as used by the forecasting model, e.g. 'SIOUX'.")
    lookback_days: int = Field(
        default=60,
        ge=1,
        le=365,
        description="Number of days of history to retrieve, counting back from today.",
    )


class HistoricalSalesPoint(BaseModel):
    date: str    # "YYYY-MM-DD"
    county: str
    value: float


@tool(args_schema=HistoricalSalesInput)
def get_historical_sales(county: str, lookback_days: int = 60) -> list[HistoricalSalesPoint]:
    """
    Retrieve daily actual sales totals for a county over a recent lookback window.
    Use this to answer questions about past sales performance, and to provide
    context (recent actuals) alongside a forecast chart.
    """
    today = date.today()
    start_date = today - timedelta(days=lookback_days)

    # Use real BigQuery if available, otherwise stub data
    if bq_client is not None:
        return _get_historical_sales_bigquery(county, start_date, today)
    else:
        return _get_historical_sales_stub(county, lookback_days)


def _get_historical_sales_bigquery(county: str, start_date: date, end_date: date) -> list[HistoricalSalesPoint]:
    """Query real BigQuery for historical sales."""
    start_time = time.time()
    input_args = {
        "county": county,
        "lookback_days": (date.today() - start_date).days,
    }

    query = f"""
    SELECT
        date,
        county,
        SUM(sale_dollars) AS value
    FROM `{BIGQUERY_DATASET}.{BIGQUERY_TABLE}`
    WHERE county = @county
        AND date BETWEEN @start_date AND @end_date
    GROUP BY date, county
    ORDER BY date ASC
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("county", "STRING", county),
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date.isoformat()),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date.isoformat()),
        ]
    )

    try:
        results = bq_client.query(query, job_config=job_config).result()
        points = [
            HistoricalSalesPoint(
                date=str(row.date),
                county=row.county,
                value=float(row.value),
            )
            for row in results
        ]
        latency_ms = (time.time() - start_time) * 1000

        # Log to observability (if available)
        try:
            from observability import observer
            observer.log_tool_execution(
                tool_name="get_historical_sales",
                input_args=input_args,
                output=points,
                latency_ms=latency_ms,
            )
        except ImportError:
            pass

        logger.info(
            f"BigQuery: Retrieved {len(points)} points for {county} "
            f"({start_date} to {end_date}) in {latency_ms:.0f}ms"
        )
        return points
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(f"BigQuery error for {county}: {e}")

        # Log error to observability
        try:
            from observability import observer
            observer.log_tool_execution(
                tool_name="get_historical_sales",
                input_args=input_args,
                output=None,
                latency_ms=latency_ms,
                error=str(e),
            )
        except ImportError:
            pass

        raise


def _get_historical_sales_stub(county: str, lookback_days: int) -> list[HistoricalSalesPoint]:
    """Fallback stub data (used if BigQuery not available)."""
    today = date.today()
    points = []
    for i in range(lookback_days, 0, -1):
        current_date = today - timedelta(days=i)
        # Stub data: vary by day to show a pattern
        base_value = 30000 + (i * 50) + (i % 7) * 2000
        points.append(
            HistoricalSalesPoint(
                date=current_date.isoformat(),
                county=county,
                value=float(base_value),
            )
        )
    return points


class ForecastInput(BaseModel):
    county: str = Field(..., description="County name as used by the forecasting model, e.g. 'SIOUX'.")
    horizon_days: int = Field(
        ...,
        ge=1,
        le=30,
        description="Number of days ahead of today to forecast. Must be between 1 and 30.",
    )


class ForecastPointOut(BaseModel):
    date: str    # "YYYY-MM-DD"
    value: float


def _check_historical_data_available(county: str, lookback_days: int = 30) -> bool:
    """Check if BigQuery has historical data for the given county and lookback period."""
    if bq_client is None:
        # If no BigQuery client, assume data is available (stub mode)
        return True

    today = date.today()
    start_date = today - timedelta(days=lookback_days)

    try:
        query = f"""
        SELECT COUNT(*) as count
        FROM `{BIGQUERY_DATASET}.{BIGQUERY_TABLE}`
        WHERE county = @county
            AND date BETWEEN DATE_SUB(PARSE_DATE('%Y-%m-%d', @from_date), INTERVAL 62 DAY) AND DATE_SUB(PARSE_DATE('%Y-%m-%d', @to_date), INTERVAL 30 DAY)
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("county", "STRING", county),
                bigquery.ScalarQueryParameter("start_date", "DATE", start_date.isoformat()),
                bigquery.ScalarQueryParameter("end_date", "DATE", today.isoformat()),
            ]
        )

        results = bq_client.query(query, job_config=job_config).result()
        row = next(results)
        has_data = row.count > 0

        logger.info(
            f"Historical data check for {county}: {row.count} records found "
            f"in {lookback_days}-day lookback"
        )
        return has_data

    except Exception as e:
        logger.warning(f"Error checking historical data availability: {e}")
        # If check fails, assume data is available to proceed with forecast
        return True


@tool(args_schema=ForecastInput)
def get_forecast(county: str, horizon_days: int) -> list[ForecastPointOut]:
    """
    Get a daily sales forecast for a county, starting tomorrow and extending
    `horizon_days` days ahead. Always returns one value per day in the horizon —
    never a partial or summarized result. Makes a single HTTP call to the
    forecasting model's /predict_array endpoint.

    First checks if historical data is available in BigQuery for context.
    """
    start_time = time.time()
    input_args = {"county": county, "horizon_days": horizon_days}

    # Check if we have historical data available for this county
    if not _check_historical_data_available(county, lookback_days=30):
        error_msg = (
            f"No recent historical sales data found for {county} in the past 30 days. "
            f"The forecasting model requires recent context to make accurate predictions. "
            f"Please ensure data is available or check the county name."
        )
        logger.warning(error_msg)
        raise ValueError(error_msg)

    today = date.today()
    from_date = today + timedelta(days=1)
    to_date = today + timedelta(days=horizon_days)

    try:
        # Call the model service (synchronously)
        with httpx.Client() as client:
            response = client.post(
                MODEL_ENDPOINT,
                json={
                    "from_date": from_date.isoformat(),
                    "to_date": to_date.isoformat(),
                    "county": county,
                },
                timeout=15.0,
            )
            response.raise_for_status()
            body = response.json()

        # Extract forecast values
        values = body.get("forecast", [])
        expected_len = (to_date - from_date).days + 1

        # Validate array length matches requested range (Section 5.4)
        if len(values) != expected_len:
            logger.error(
                f"Model service returned {len(values)} values, expected {expected_len}"
            )
            raise ValueError(
                f"Forecast length mismatch: expected {expected_len} values "
                f"for {from_date} to {to_date}, got {len(values)}. "
                "Refusing to guess at date alignment."
            )

        # Map forecast values to dates (Section 6.2.1: positional mapping)
        points = [
            ForecastPointOut(
                date=(from_date + timedelta(days=i)).isoformat(),
                value=float(v),
            )
            for i, v in enumerate(values)
        ]

        latency_ms = (time.time() - start_time) * 1000

        # Log to observability (if available)
        try:
            from observability import observer
            observer.log_tool_execution(
                tool_name="get_forecast",
                input_args=input_args,
                output=values,
                latency_ms=latency_ms,
            )
        except ImportError:
            pass

        logger.info(
            f"Model service: Retrieved {len(points)} forecast points for {county} "
            f"({horizon_days} days) in {latency_ms:.0f}ms"
        )
        return points

    except httpx.HTTPStatusError as e:
        latency_ms = (time.time() - start_time) * 1000
        error_msg = f"Model service returned {e.response.status_code}"
        logger.error(f"{error_msg}: {e.response.text}")

        # Log error
        try:
            from observability import observer
            observer.log_tool_execution(
                tool_name="get_forecast",
                input_args=input_args,
                output=None,
                latency_ms=latency_ms,
                error=error_msg,
            )
        except ImportError:
            pass

        raise Exception(f"{error_msg}: {e.response.text}")

    except httpx.RequestError as e:
        latency_ms = (time.time() - start_time) * 1000
        error_msg = f"Failed to connect to model service at {MODEL_ENDPOINT}"
        logger.error(f"{error_msg}: {e}")

        # Log error
        try:
            from observability import observer
            observer.log_tool_execution(
                tool_name="get_forecast",
                input_args=input_args,
                output=None,
                latency_ms=latency_ms,
                error=error_msg,
            )
        except ImportError:
            pass

        raise Exception(error_msg)

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(f"Forecast error: {e}")

        # Log error
        try:
            from observability import observer
            observer.log_tool_execution(
                tool_name="get_forecast",
                input_args=input_args,
                output=None,
                latency_ms=latency_ms,
                error=str(e),
            )
        except ImportError:
            pass

        raise
