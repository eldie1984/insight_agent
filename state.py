from typing import Annotated, Literal, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class ForecastPoint(TypedDict):
    """A single data point for charting (actual or forecast value)."""
    date: str          # ISO-8601 "YYYY-MM-DD"
    value: float
    type: Literal["actual", "forecast"]


class SalesAssistantState(TypedDict):
    """State schema for the Sales Assistant agent graph."""
    messages: Annotated[list[AnyMessage], add_messages]

    # Populated by tool execution / chart builder; consumed by the response payload.
    chart_data: list[ForecastPoint] | None
    chart_meta: dict | None          # {"county": str, "horizon_days": int, "generated_at": str}

    # Set during date validation (Section 5); used to short-circuit malformed requests
    # before any tool call is attempted.
    resolved_horizon_days: int | None
    validation_error: str | None
