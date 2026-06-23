"""
Test suite for the LangGraph skeleton.

Validates routing logic (route_after_agent, route_after_tools)
and chart_builder node in isolation using stub tools.
"""
import pytest
from datetime import date
from unittest.mock import MagicMock, AsyncMock
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

from graph import (
    validate_request_node,
    route_after_agent,
    route_after_tools,
    chart_builder_node,
    MAX_HORIZON_DAYS,
)
from state import SalesAssistantState, ForecastPoint


class TestValidateRequestNode:
    """Test the validate_request node's horizon validation."""

    def test_no_candidate_passes_through(self):
        """When no resolved_horizon_days is set, should return empty dict."""
        state = {
            "messages": [],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": None,
            "validation_error": None,
        }
        result = validate_request_node(state)
        assert result == {}

    def test_negative_horizon_rejected(self):
        """Negative horizon should be rejected."""
        state = {
            "messages": [],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": -1,
            "validation_error": None,
        }
        result = validate_request_node(state)
        assert result["resolved_horizon_days"] is None
        assert "at least 1 day" in result["validation_error"].lower()

    def test_zero_horizon_rejected(self):
        """Zero horizon should be rejected."""
        state = {
            "messages": [],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": 0,
            "validation_error": None,
        }
        result = validate_request_node(state)
        assert result["resolved_horizon_days"] is None
        assert "at least 1 day" in result["validation_error"].lower()

    def test_valid_horizon_accepted(self):
        """Valid horizon (1-30) should pass."""
        for horizon in [1, 15, 30]:
            state = {
                "messages": [],
                "chart_data": None,
                "chart_meta": None,
                "resolved_horizon_days": horizon,
                "validation_error": None,
            }
            result = validate_request_node(state)
            assert result["resolved_horizon_days"] == horizon
            assert "validation_error" not in result or result.get("validation_error") is None

    def test_horizon_over_max_rejected(self):
        """Horizon > 30 days should be rejected."""
        state = {
            "messages": [],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": 31,
            "validation_error": None,
        }
        result = validate_request_node(state)
        assert result["resolved_horizon_days"] is None
        assert "30 days" in result["validation_error"]

    def test_horizon_way_over_max_rejected(self):
        """Very large horizon should be rejected."""
        state = {
            "messages": [],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": 100,
            "validation_error": None,
        }
        result = validate_request_node(state)
        assert result["resolved_horizon_days"] is None
        assert MAX_HORIZON_DAYS in str(result["validation_error"])


class TestRouteAfterAgent:
    """Test the routing logic after the agent node."""

    def test_routes_to_tools_when_tool_calls_present(self):
        """Should route to 'tools' when AIMessage has tool_calls."""
        ai_message = AIMessage(content="Calling a tool", tool_calls=[{"id": "1", "function": "forecast"}])
        state = {
            "messages": [ai_message],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": None,
            "validation_error": None,
        }
        result = route_after_agent(state)
        assert result == "tools"

    def test_routes_to_end_when_no_tool_calls(self):
        """Should route to 'end' when no tool_calls in AIMessage."""
        ai_message = AIMessage(content="Here is the answer to your question.")
        state = {
            "messages": [ai_message],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": None,
            "validation_error": None,
        }
        result = route_after_agent(state)
        assert result == "end"

    def test_routes_to_end_when_message_has_empty_tool_calls(self):
        """Should route to 'end' when tool_calls list is empty."""
        ai_message = AIMessage(content="Answer", tool_calls=[])
        state = {
            "messages": [ai_message],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": None,
            "validation_error": None,
        }
        result = route_after_agent(state)
        assert result == "end"


class TestRouteAfterTools:
    """Test the routing logic after the tools node."""

    def test_routes_to_chart_builder_when_forecast_tool_called(self):
        """Should route to 'chart_builder' when get_forecast was called."""
        forecast_output = [
            {"date": "2026-06-20", "value": 40000.0},
            {"date": "2026-06-21", "value": 41000.0},
        ]
        tool_message = ToolMessage(
            content=str(forecast_output),
            tool_call_id="forecast_call_1",
            name="get_forecast",
        )
        state = {
            "messages": [tool_message],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": None,
            "validation_error": None,
        }
        result = route_after_tools(state)
        assert result == "chart_builder"

    def test_routes_to_agent_when_only_historical_called(self):
        """Should route back to 'agent' when only historical sales was called."""
        historical_output = [
            {"date": "2026-06-19", "county": "SIOUX", "value": 30000.0},
        ]
        tool_message = ToolMessage(
            content=str(historical_output),
            tool_call_id="historical_call_1",
            name="get_historical_sales",
        )
        state = {
            "messages": [tool_message],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": None,
            "validation_error": None,
        }
        result = route_after_tools(state)
        assert result == "agent"

    def test_routes_to_chart_builder_when_forecast_in_name(self):
        """Should match 'forecast' in tool name (case-insensitive)."""
        tool_message = ToolMessage(
            content="[{'date': '2026-06-20', 'value': 40000}]",
            tool_call_id="call",
            name="GetForecast",  # Different casing
        )
        state = {
            "messages": [tool_message],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": None,
            "validation_error": None,
        }
        result = route_after_tools(state)
        assert result == "chart_builder"

    def test_ignores_human_messages_when_routing(self):
        """Should skip HumanMessages and only check ToolMessages."""
        human_msg = HumanMessage(content="Some human message")
        tool_message = ToolMessage(
            content="[{'date': '2026-06-20', 'value': 40000}]",
            tool_call_id="call",
            name="get_forecast",
        )
        state = {
            "messages": [human_msg, tool_message],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": None,
            "validation_error": None,
        }
        result = route_after_tools(state)
        assert result == "chart_builder"


class TestChartBuilderNode:
    """Test the chart_builder node's logic."""

    def test_builds_chart_from_forecast_only(self):
        """Chart should be built from forecast points."""
        forecast_output = [
            {"date": "2026-06-20", "value": 40000.0},
            {"date": "2026-06-21", "value": 41000.0},
            {"date": "2026-06-22", "value": 42000.0},
        ]
        tool_message = ToolMessage(
            content=str(forecast_output),
            tool_call_id="forecast_call",
            name="get_forecast",
        )
        state = {
            "messages": [tool_message],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": None,
            "validation_error": None,
        }
        result = chart_builder_node(state)

        assert result["chart_data"] is not None
        assert len(result["chart_data"]) == 3
        assert all(pt["type"] == "forecast" for pt in result["chart_data"])
        assert result["chart_meta"] is not None
        assert result["chart_meta"]["horizon_days"] == 3

    def test_prepends_historical_to_forecast(self):
        """Historical actuals should come before forecast in chart_data."""
        historical_output = [
            {"date": "2026-06-18", "county": "SIOUX", "value": 30000.0},
            {"date": "2026-06-19", "county": "SIOUX", "value": 31000.0},
        ]
        forecast_output = [
            {"date": "2026-06-20", "value": 40000.0},
            {"date": "2026-06-21", "value": 41000.0},
        ]
        historical_msg = ToolMessage(
            content=str(historical_output),
            tool_call_id="historical_call",
            name="get_historical_sales",
        )
        forecast_msg = ToolMessage(
            content=str(forecast_output),
            tool_call_id="forecast_call",
            name="get_forecast",
        )
        state = {
            "messages": [historical_msg, forecast_msg],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": None,
            "validation_error": None,
        }
        result = chart_builder_node(state)

        assert result["chart_data"] is not None
        assert len(result["chart_data"]) == 4
        # First two should be actual, last two should be forecast
        assert all(pt["type"] == "actual" for pt in result["chart_data"][:2])
        assert all(pt["type"] == "forecast" for pt in result["chart_data"][2:])

    def test_no_chart_without_forecast(self):
        """Chart should not be created without forecast points."""
        historical_output = [
            {"date": "2026-06-19", "county": "SIOUX", "value": 31000.0},
        ]
        historical_msg = ToolMessage(
            content=str(historical_output),
            tool_call_id="historical_call",
            name="get_historical_sales",
        )
        state = {
            "messages": [historical_msg],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": None,
            "validation_error": None,
        }
        result = chart_builder_node(state)

        assert result["chart_data"] is None
        assert result["chart_meta"] is None

    def test_no_chart_when_no_tool_messages(self):
        """Chart should not be created when there are no ToolMessages."""
        human_msg = HumanMessage(content="What is the forecast?")
        state = {
            "messages": [human_msg],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": None,
            "validation_error": None,
        }
        result = chart_builder_node(state)

        assert result["chart_data"] is None
        assert result["chart_meta"] is None

    def test_correct_date_values_in_chart(self):
        """Chart points should have correct date and value fields."""
        forecast_output = [
            {"date": "2026-06-20", "value": 40000.0},
            {"date": "2026-06-21", "value": 41000.0},
        ]
        tool_message = ToolMessage(
            content=str(forecast_output),
            tool_call_id="call",
            name="get_forecast",
        )
        state = {
            "messages": [tool_message],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": None,
            "validation_error": None,
        }
        result = chart_builder_node(state)

        assert result["chart_data"][0]["date"] == "2026-06-20"
        assert result["chart_data"][0]["value"] == 40000.0
        assert result["chart_data"][1]["date"] == "2026-06-21"
        assert result["chart_data"][1]["value"] == 41000.0

    def test_chart_meta_has_required_fields(self):
        """Chart metadata should include county, horizon_days, and generated_at."""
        forecast_output = [
            {"date": "2026-06-20", "value": 40000.0},
            {"date": "2026-06-21", "value": 41000.0},
        ]
        tool_message = ToolMessage(
            content=str(forecast_output),
            tool_call_id="call",
            name="get_forecast",
        )
        state = {
            "messages": [tool_message],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": None,
            "validation_error": None,
        }
        result = chart_builder_node(state)

        assert result["chart_meta"] is not None
        assert "horizon_days" in result["chart_meta"]
        assert result["chart_meta"]["horizon_days"] == 2
        assert "generated_at" in result["chart_meta"]
        assert "county" in result["chart_meta"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
