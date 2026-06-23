"""
LangGraph graph skeleton for the Sales Assistant agent.

Implements the graph topology from Section 4 of the SDD:
- validate_request: enforce 30-day horizon rule
- agent: LLM with tool calling
- tools: execute requested tools
- chart_builder: convert tool outputs to chart data
- Conditional routing after agent and tools nodes
"""
from datetime import date
from typing import Literal
import json
import logging
import calendar

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI

from state import SalesAssistantState, ForecastPoint
from tools import get_historical_sales, get_forecast
from config import settings

logger = logging.getLogger(__name__)


MAX_HORIZON_DAYS = 30


def get_days_remaining_in_month(target_date: date) -> int:
    """Calculate days remaining in the month from target_date."""
    last_day = calendar.monthrange(target_date.year, target_date.month)[1]
    return last_day - target_date.day + 1


def validate_request_node(state: SalesAssistantState) -> dict:
    """
    Enforce the 30-day-ahead rule before any tool call happens.

    Runs before agent on every new human turn. Sets resolved_horizon_days
    and/or validation_error for the agent to handle.
    """
    candidate = state.get("resolved_horizon_days")

    if candidate is None:
        # Nothing to validate yet — fall through to agent
        return {}

    if candidate <= 0:
        return {
            "validation_error": "Horizon must be at least 1 day ahead of today.",
            "resolved_horizon_days": None,
        }

    if candidate > MAX_HORIZON_DAYS:
        return {
            "validation_error": (
                f"The forecasting model supports up to {MAX_HORIZON_DAYS} days ahead. "
                f"Please ask for a horizon of {MAX_HORIZON_DAYS} days or fewer."
            ),
            "resolved_horizon_days": None,
        }

    return {"resolved_horizon_days": candidate}


def agent_node(state: SalesAssistantState) -> dict:
    """
    Single LLM call with tool-calling enabled.

    Decides whether to answer directly, call a tool, or ask a clarifying question.
    """
    # Reset chart data at the start of each new user turn
    chart_data = None
    chart_meta = None

    # If validation failed, inject the error as a system correction
    messages = list(state["messages"])
    if state.get("validation_error"):
        error_msg = ToolMessage(
            content=f"Validation error: {state['validation_error']}",
            tool_call_id="validation",
        )
        messages = messages + [error_msg]

    # Build system prompt with today's date
    current_date = date.today()
    days_left_this_month = get_days_remaining_in_month(current_date)

    system_prompt = f"""You are a sales forecasting assistant for internal business users. Be concise, avoid technical jargon.

Key facts:
- Today's date is {current_date.isoformat()} ({current_date.strftime('%B %d, %Y')}).
- The forecasting model always forecasts forward from today for a number of days you specify (max 30 days).
- The model requires a county (e.g. "SIOUX"). If not specified, ask.
- When asked for a forecast, also fetch recent historical sales for context.
- Always report every day's value for the requested horizon — never truncate or sample.
- When your answer includes forecasted numbers, a chart is built automatically.

Date interpretation guide:
- "30 days" → horizon_days = 30
- "next week" → horizon_days = 7
- "this month" or "rest of month" → horizon_days = {days_left_this_month} (days left in {current_date.strftime('%B')})
- "next month" → horizon_days = 30
- "the month of [past month]" → Ask user to clarify (already passed)
- "[future month name]" where > 30 days away → Explain model limit and offer alternative (30 days)
- Always validate horizon_days is between 1 and 30 before calling get_forecast tool."""

    # Prepare for tool calling
    tools = [get_historical_sales, get_forecast]

    # Validate OpenRouter API key
    if not settings.openrouter_api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not set. Get it from https://openrouter.ai/keys"
        )

    # Call the LLM with tool-calling enabled (OpenRouter -> Claude)
    llm = ChatOpenAI(
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=0,
    )

    # Bind tools - OpenRouter's Claude handles this perfectly
    llm_with_tools = llm.bind_tools(tools)

    # Prepare messages with system prompt at the beginning
    messages_with_system = [SystemMessage(content=system_prompt)] + messages

    # Invoke LLM with messages including system prompt
    response = llm_with_tools.invoke(messages_with_system)

    logger.info(f"Agent response: {response}")
    logger.info(f"Agent has tool_calls: {hasattr(response, 'tool_calls') and bool(response.tool_calls)}")
    if hasattr(response, 'tool_calls') and response.tool_calls:
        logger.info(f"Tool calls: {response.tool_calls}")

    return {
        "messages": [response],
        "chart_data": chart_data,
        "chart_meta": chart_meta,
        "validation_error": None,
    }


def route_after_agent(state: SalesAssistantState) -> Literal["tools", "end"]:
    """
    Route after agent node: check if LLM requested any tool calls.
    """
    last_message = state["messages"][-1]
    has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls
    logger.info(f"route_after_agent: has_tool_calls={has_tool_calls}, message_type={type(last_message).__name__}")
    if has_tool_calls:
        logger.info(f"  tool_calls: {last_message.tool_calls}")
        return "tools"
    logger.info(f"  content: {last_message.content if hasattr(last_message, 'content') else 'N/A'}")
    return "end"


def route_after_tools(state: SalesAssistantState) -> Literal["chart_builder", "end"]:
    """
    Route after tools node: always go to chart_builder to extract and format data.
    Never loop back to agent after tools - that causes tool_call_id validation errors.
    """
    logger.info(f"route_after_tools: Total messages = {len(state['messages'])}")
    for i, msg in enumerate(state["messages"]):
        if isinstance(msg, ToolMessage):
            logger.info(f"  Message {i}: ToolMessage, name={msg.name}")
        else:
            logger.info(f"  Message {i}: {type(msg).__name__}")

    # Always route to chart_builder after tools execute
    return "chart_builder"


def chart_builder_node(state: SalesAssistantState) -> dict:
    """
    Convert raw tool outputs (historical rows + forecast rows) into chart data.

    Scans recent ToolMessages for outputs from get_forecast and get_historical_sales,
    then builds the ForecastPoint[] structure for the response payload.
    """
    chart_data = None
    chart_meta = None

    # Collect recent tool outputs
    forecast_points = []
    historical_points = []

    for message in reversed(state["messages"]):
        if not isinstance(message, ToolMessage):
            continue

        # Parse the tool output
        try:
            if isinstance(message.content, str):
                # Try to parse JSON
                content = json.loads(message.content)
            else:
                content = message.content

            # Check tool type by name
            if "forecast" in (message.name or "").lower():
                # get_forecast output: list of {date, value}
                if isinstance(content, list):
                    for point in content:
                        forecast_points.append(point)
            elif "historical" in (message.name or "").lower():
                # get_historical_sales output: list of {date, county, value}
                if isinstance(content, list):
                    for point in content:
                        historical_points.append(point)
        except (json.JSONDecodeError, TypeError):
            # If content can't be parsed, skip
            pass

    # Build chart_data if we have forecast points
    if forecast_points:
        chart_data = []

        # Add historical points first (if available)
        for point in historical_points:
            chart_data.append(
                ForecastPoint(
                    date=point.get("date"),
                    value=float(point.get("value")),
                    type="actual",
                )
            )

        # Add forecast points
        for point in forecast_points:
            chart_data.append(
                ForecastPoint(
                    date=point.get("date"),
                    value=float(point.get("value")),
                    type="forecast",
                )
            )

        # Build metadata
        if forecast_points:
            horizon_days = len(forecast_points)
            chart_meta = {
                "county": state["messages"][-1].content.get("county", "UNKNOWN") if hasattr(state["messages"][-1], "content") else "UNKNOWN",
                "horizon_days": horizon_days,
                "generated_at": date.today().isoformat() + "T00:00:00Z",
            }

    return {
        "chart_data": chart_data,
        "chart_meta": chart_meta,
    }


def build_graph() -> StateGraph:
    """Construct and return the compiled LangGraph graph."""
    graph = StateGraph(SalesAssistantState)

    # Add nodes
    graph.add_node("validate_request", validate_request_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode([get_historical_sales, get_forecast]))
    graph.add_node("chart_builder", chart_builder_node)

    # Set entry point
    graph.set_entry_point("validate_request")

    # Add edges
    graph.add_edge("validate_request", "agent")

    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "end": END},
    )

    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {"chart_builder": "chart_builder", "end": END},
    )

    graph.add_edge("chart_builder", END)

    return graph.compile()
