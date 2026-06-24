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
from typing import Literal, Any
import json
import logging
import calendar

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage, SystemMessage, AIMessage
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
    Preserves chart_data and chart_meta if already set (from chart_builder).
    """
    # Preserve chart data if it exists (from chart_builder)
    chart_data = state.get("chart_data")
    chart_meta = state.get("chart_meta")

    # If validation failed, inject the error as a system correction
    messages = list(state["messages"])
    if state.get("validation_error"):
        error_msg = ToolMessage(
            content=f"Validation error: {state['validation_error']}",
            tool_call_id="validation",
        )
        messages = messages + [error_msg]

    # Check if only forecast has been called (no historical)
    has_forecast_call = False
    has_historical_call = False
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            if "forecast" in (msg.name or "").lower():
                has_forecast_call = True
            elif "historical" in (msg.name or "").lower():
                has_historical_call = True

    # If we have forecast but missing historical, append instruction to system prompt
    forecast_missing_historical = has_forecast_call and not has_historical_call
    if forecast_missing_historical:
        logger.info(
            "Detected forecast without historical data - will instruct agent to call get_historical_sales"
        )

    # Build system prompt with today's date
    current_date = date.today()
    days_left_this_month = get_days_remaining_in_month(current_date)

    system_prompt = f"""You are a sales forecasting assistant for internal business users. Be concise, avoid technical jargon.

Key facts:
- Today's date is {current_date.isoformat()} ({current_date.strftime("%B %d, %Y")}).
- The forecasting model always forecasts forward from today for a number of days you specify (max 30 days).
- CRITICAL: County MUST be explicitly mentioned in the CURRENT message to proceed.
  - Examples of VALID requests (with county):
    - "Forecast for SIOUX for 30 days"
    - "What's the forecast for LYON county?"
    - "Show me SIOUX sales for next month"
  - Examples of INVALID requests (without county - MUST ASK):
    - "What would be the forecast for march?" → ASK: "Which county?"
    - "Show me a 30 day forecast" → ASK: "Which county would you like?"
  - Do NOT call any tools without an explicit county in the current message
  - Do NOT assume or infer a county from previous messages
- ONLY call tools when county is explicitly mentioned in current message:
  1. get_historical_sales(county, lookback_days=60) - to show recent actual sales context
  2. get_forecast(county, horizon_days=X, from_date=optional) - to get predictions
- This creates a chart with both actual (green) and forecast (blue) data for comparison
- Always report every day's value for the requested horizon — never truncate or sample.
- When your answer includes forecasted numbers, a chart is built automatically with both series.

Date interpretation guide:
- FORECAST REQUESTS (future dates):
  - "30 days" → horizon_days = 30
  - "next week" → horizon_days = 7
  - "this month" or "rest of month" → horizon_days = {days_left_this_month} (days left in {current_date.strftime("%B")})
  - "next month" → horizon_days = 30
  - Validate horizon_days is between 1 and 30

- HISTORICAL FORECAST VALIDATION (past dates):
  - If user asks "what was the forecast for May?" → COMPARE forecast predictions vs actual values
  - Two-step approach:
    1. Call get_forecast(county, horizon_days=31, from_date="2026-05-01") - what model predicted
    2. Call get_historical_sales(county, lookback_days=60) - what actually happened
  - Return BOTH in chart with dual series: "forecast" type (blue) vs "actual" type (green)
  - Show the forecast accuracy: how close was the prediction to reality

- CLARIFICATION:
  - Forecasts = future predictions OR historical predictions for validation
  - Historical = actual past sales data (from BigQuery)
  - from_date parameter enables "what did the model predict for [past period]?" analysis
  - This shows model accuracy and performance over time
"""

    # If we only have forecast and need historical, add explicit instruction
    if forecast_missing_historical:
        system_prompt += "\n\nIMPORTANT: You already have forecast data. Now MUST call get_historical_sales to get the actual historical sales data so we can show both forecast and actual values in the chart for comparison."

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
        api_key=settings.openrouter_api_key,  # type: ignore[arg-type]
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
    logger.info(
        f"Agent has tool_calls: {hasattr(response, 'tool_calls') and bool(response.tool_calls)}"
    )
    if hasattr(response, "tool_calls") and response.tool_calls:
        logger.info(f"Tool calls: {response.tool_calls}")

    return {
        "messages": [response],
        "chart_data": chart_data,
        "chart_meta": chart_meta,
        "validation_error": None,
        "needs_historical_data": False,  # Reset the flag
    }


def route_after_agent(state: SalesAssistantState) -> Literal["tools", "end"]:
    """
    Route after agent node: check if LLM requested any tool calls.
    """
    last_message = state["messages"][-1]
    has_tool_calls = (
        isinstance(last_message, AIMessage)
        and hasattr(last_message, "tool_calls")
        and bool(last_message.tool_calls)  # type: ignore[union-attr]
    )
    logger.info(
        f"route_after_agent: has_tool_calls={has_tool_calls}, message_type={type(last_message).__name__}"
    )
    if has_tool_calls:
        logger.info(f"  tool_calls: {last_message.tool_calls}")  # type: ignore[union-attr]
        return "tools"
    logger.info(
        f"  content: {last_message.content if hasattr(last_message, 'content') else 'N/A'}"
    )
    return "end"


def route_after_tools(state: SalesAssistantState) -> Literal["chart_builder", "agent"]:
    """
    Route after tools node:
    - If only forecast was called, route back to agent to call get_historical_sales
    - If both were called, route to chart_builder
    """
    logger.info(f"route_after_tools: Total messages = {len(state['messages'])}")

    has_forecast = False
    has_historical = False

    for i, msg in enumerate(state["messages"]):
        if isinstance(msg, ToolMessage):
            logger.info(f"  Message {i}: ToolMessage, name={msg.name}")
            if "forecast" in (msg.name or "").lower():
                has_forecast = True
            elif "historical" in (msg.name or "").lower():
                has_historical = True
        else:
            logger.info(f"  Message {i}: {type(msg).__name__}")

    logger.info(f"  has_forecast={has_forecast}, has_historical={has_historical}")

    # Return dict with routing decision AND state update if needed
    if has_forecast and not has_historical:
        logger.info("  → Routing back to agent to call get_historical_sales")
        # Will be handled as a state update in the agent node
        return "agent"

    # Otherwise, we have all the data we need
    logger.info("  → Routing to chart_builder")
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

    logger.info(f"chart_builder: Processing {len(state['messages'])} messages")

    for message in reversed(state["messages"]):
        if not isinstance(message, ToolMessage):
            continue

        logger.info(
            f"Found ToolMessage: name={message.name}, content_len={len(str(message.content))}"
        )

        # Parse the tool output
        try:
            if isinstance(message.content, str):
                # Try to parse JSON
                content = json.loads(message.content)
            else:
                content = message.content

            logger.info(f"  Parsed content type: {type(content).__name__}")

            # Check tool type by name
            if "forecast" in (message.name or "").lower():
                logger.info(
                    f"  → Found FORECAST tool, {len(content) if isinstance(content, list) else 0} points"
                )
                # get_forecast output: list of {date, value}
                if isinstance(content, list):
                    for point in content:
                        forecast_points.append(point)
            elif "historical" in (message.name or "").lower():
                logger.info(
                    f"  → Found HISTORICAL tool, {len(content) if isinstance(content, list) else 0} points"
                )
                # get_historical_sales output: list of {date, county, value}
                if isinstance(content, list):
                    for point in content:
                        historical_points.append(point)
            else:
                logger.info("  → Unknown tool type, skipping")
        except (json.JSONDecodeError, TypeError) as e:
            # If content can't be parsed, skip
            logger.warning(f"  Could not parse content: {e}")

    logger.info(
        f"chart_builder: Found {len(forecast_points)} forecast points, {len(historical_points)} historical points"
    )

    # Build chart_data if we have forecast points
    if not forecast_points:
        return {"chart_data": None, "chart_meta": None}

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

    # Build metadata - extract county from AI message tool calls
    horizon_days = len(forecast_points)
    county = "UNKNOWN"

    # Search for county in AI message tool calls
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call.get("args", {}).get("county"):
                    county = tool_call["args"]["county"]
                    break
            if county != "UNKNOWN":
                break

    chart_meta = {
        "county": county,
        "horizon_days": horizon_days,
        "generated_at": date.today().isoformat() + "T00:00:00Z",
    }

    return {
        "chart_data": chart_data,
        "chart_meta": chart_meta,
    }


def build_graph() -> Any:
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
        {"chart_builder": "chart_builder", "agent": "agent"},
    )

    # After chart_builder, route back to agent for final summary
    graph.add_edge("chart_builder", "agent")

    return graph.compile()
