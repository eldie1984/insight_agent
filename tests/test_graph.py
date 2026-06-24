"""Tests for LangGraph agent graph."""

from datetime import date
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from graph import (
    validate_request_node,
    route_after_agent,
    route_after_tools,
    chart_builder_node,
    get_days_remaining_in_month,
)
from state import SalesAssistantState


class TestValidateRequestNode:
    """Test request validation."""

    def test_passes_through_when_no_horizon(self):
        """Should return empty dict when no horizon to validate."""
        state: SalesAssistantState = {
            "messages": [],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": None,
            "validation_error": None,
            "needs_historical_data": False,
        }
        result = validate_request_node(state)
        assert result == {}

    def test_rejects_zero_or_negative_horizon(self):
        """Should reject horizon <= 0."""
        state: SalesAssistantState = {
            "messages": [],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": 0,
            "validation_error": None,
            "needs_historical_data": False,
        }
        result = validate_request_node(state)
        assert "validation_error" in result
        assert "at least 1 day" in result["validation_error"]

    def test_rejects_horizon_over_30_days(self):
        """Should reject horizon > 30 days."""
        state: SalesAssistantState = {
            "messages": [],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": 31,
            "validation_error": None,
            "needs_historical_data": False,
        }
        result = validate_request_node(state)
        assert "validation_error" in result
        assert "30 days" in result["validation_error"]

    def test_accepts_valid_horizon(self):
        """Should accept horizon between 1-30 days."""
        for days in [1, 15, 30]:
            state: SalesAssistantState = {
                "messages": [],
                "chart_data": None,
                "chart_meta": None,
                "resolved_horizon_days": days,
                "validation_error": None,
                "needs_historical_data": False,
            }
            result = validate_request_node(state)
            assert result == {"resolved_horizon_days": days}


class TestRouteAfterAgent:
    """Test agent routing logic."""

    def test_routes_to_tools_when_tool_calls_present(self):
        """Should route to tools when agent makes tool calls."""
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_forecast",
                    "args": {"county": "SIOUX", "horizon_days": 30},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        )
        state: SalesAssistantState = {
            "messages": [HumanMessage(content="test"), ai_msg],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": None,
            "validation_error": None,
            "needs_historical_data": False,
        }
        result = route_after_agent(state)
        assert result == "tools"

    def test_routes_to_end_when_no_tool_calls(self):
        """Should route to end when agent doesn't make tool calls."""
        ai_msg = AIMessage(content="Please specify a county")
        state: SalesAssistantState = {
            "messages": [HumanMessage(content="test"), ai_msg],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": None,
            "validation_error": None,
            "needs_historical_data": False,
        }
        result = route_after_agent(state)
        assert result == "end"


class TestRouteAfterTools:
    """Test tool execution routing logic."""

    def test_routes_to_chart_builder_when_both_tools_called(self):
        """Should route to chart_builder when both tools executed."""
        state: SalesAssistantState = {
            "messages": [
                HumanMessage(content="test"),
                AIMessage(content="", tool_calls=[]),
                ToolMessage(
                    name="get_historical_sales", content="[]", tool_call_id="1"
                ),
                ToolMessage(name="get_forecast", content="[]", tool_call_id="2"),
            ],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": None,
            "validation_error": None,
            "needs_historical_data": False,
        }
        result = route_after_tools(state)
        assert result == "chart_builder"

    def test_routes_to_agent_when_only_forecast(self):
        """Should route back to agent when only forecast (missing historical)."""
        state: SalesAssistantState = {
            "messages": [
                HumanMessage(content="test"),
                AIMessage(content="", tool_calls=[]),
                ToolMessage(name="get_forecast", content="[]", tool_call_id="1"),
            ],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": None,
            "validation_error": None,
            "needs_historical_data": False,
        }
        result = route_after_tools(state)
        assert result == "agent"


class TestGetDaysRemainingInMonth:
    """Test month duration calculation."""

    def test_calculates_remaining_days_correctly(self):
        """Should calculate days from date to end of month."""
        # June 24 → 7 days left (25, 26, 27, 28, 29, 30)
        remaining = get_days_remaining_in_month(date(2026, 6, 24))
        assert remaining == 7

    def test_returns_1_on_last_day_of_month(self):
        """Should return 1 on the last day of month."""
        remaining = get_days_remaining_in_month(date(2026, 6, 30))
        assert remaining == 1

    def test_works_for_different_months(self):
        """Should work for months with different lengths."""
        # February (28 days in 2026)
        feb_remaining = get_days_remaining_in_month(date(2026, 2, 1))
        assert feb_remaining == 28

        # April (30 days)
        apr_remaining = get_days_remaining_in_month(date(2026, 4, 1))
        assert apr_remaining == 30


class TestChartBuilderNode:
    """Test chart building."""

    def test_extracts_forecast_and_historical_data(self):
        """Should extract both forecast and historical data from tool messages."""
        import json

        forecast_data = [
            {"date": "2026-03-01", "value": 100.0},
            {"date": "2026-03-02", "value": 110.0},
        ]
        historical_data = [
            {"date": "2026-02-28", "county": "SIOUX", "value": 95.0},
            {"date": "2026-02-27", "county": "SIOUX", "value": 90.0},
        ]

        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_forecast",
                    "args": {"county": "SIOUX", "horizon_days": 2},
                    "id": "call_1",
                    "type": "tool_call",
                },
                {
                    "name": "get_historical_sales",
                    "args": {"county": "SIOUX", "lookback_days": 60},
                    "id": "call_2",
                    "type": "tool_call",
                },
            ],
        )

        state: SalesAssistantState = {
            "messages": [
                HumanMessage(content="test"),
                ai_msg,
                ToolMessage(
                    name="get_forecast",
                    content=json.dumps(forecast_data),
                    tool_call_id="call_1",
                ),
                ToolMessage(
                    name="get_historical_sales",
                    content=json.dumps(historical_data),
                    tool_call_id="call_2",
                ),
            ],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": None,
            "validation_error": None,
            "needs_historical_data": False,
        }

        result = chart_builder_node(state)
        assert result["chart_data"] is not None
        assert result["chart_meta"] is not None
        assert len(result["chart_data"]) == 4  # 2 historical + 2 forecast

    def test_extracts_county_from_tool_calls(self):
        """Should extract county from AI message tool calls."""
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_forecast",
                    "args": {"county": "BOONE", "horizon_days": 2},
                    "id": "call_1",
                    "type": "tool_call",
                },
            ],
        )

        state: SalesAssistantState = {
            "messages": [
                HumanMessage(content="test"),
                ai_msg,
                ToolMessage(
                    name="get_forecast",
                    content='[{"date": "2026-03-01", "value": 100.0}]',
                    tool_call_id="call_1",
                ),
            ],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": None,
            "validation_error": None,
            "needs_historical_data": False,
        }

        result = chart_builder_node(state)
        assert result["chart_meta"]["county"] == "BOONE"

    def test_returns_none_chart_when_no_forecast(self):
        """Should return None chart_data when no forecast points."""
        state: SalesAssistantState = {
            "messages": [HumanMessage(content="test")],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": None,
            "validation_error": None,
            "needs_historical_data": False,
        }

        result = chart_builder_node(state)
        assert result["chart_data"] is None
        assert result["chart_meta"] is None
