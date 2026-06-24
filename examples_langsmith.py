"""
Examples of using LangSmith observability with the Sales Assistant Agent.

These examples demonstrate how to enable, configure, and use LangSmith
for tracing and debugging the agent.
"""

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


def example_1_enable_langsmith():
    """Example 1: Enable LangSmith tracing.

    Required:
    1. Set LANGSMITH_TRACING=true in .env
    2. Set LANGSMITH_API_KEY=lsv_... in .env
    3. Set LANGSMITH_PROJECT=sales-assistant-agent in .env
    """
    print("Example 1: Enable LangSmith Tracing")
    print("=" * 60)

    from config import settings

    if settings.langsmith_tracing:
        print("✓ LangSmith tracing enabled")
        print(f"  Project: {settings.langsmith_project}")
        print(f"  API Key: {settings.langsmith_api_key[:20]}...")
    else:
        print("✗ LangSmith tracing disabled")
        print("  Set LANGSMITH_TRACING=true in .env to enable")


def example_2_log_graph_execution():
    """Example 2: Log a graph execution to LangSmith.

    Demonstrates how to trace a complete graph run.
    """
    print("\nExample 2: Log Graph Execution")
    print("=" * 60)

    from observability import observer

    # Simulate a graph execution
    observer.log_graph_execution(
        thread_id="conv-example-001",
        input_state={"messages": 1, "chart_data": None},
        output_state={"messages": 3, "chart_data": 30},
        latency_ms=1245.3,
        nodes_executed=["validate_request", "agent", "tools", "chart_builder"],
        chart_generated=True,
    )

    print("✓ Graph execution logged")
    print("  Thread: conv-example-001")
    print("  Nodes: validate_request → agent → tools → chart_builder")
    print("  Latency: 1245.3ms")
    print("  Chart generated: Yes (30 points)")


def example_3_log_tool_execution():
    """Example 3: Log tool executions to LangSmith.

    Demonstrates tracing individual tool calls.
    """
    print("\nExample 3: Log Tool Execution")
    print("=" * 60)

    from observability import observer

    # Log a get_forecast tool call
    observer.log_tool_execution(
        tool_name="get_forecast",
        input_args={"county": "SIOUX", "horizon_days": 30},
        output=[40000.0, 41000.0, 42000.0],  # Simplified output
        latency_ms=1200.5,
    )
    print("✓ Tool execution logged: get_forecast")
    print("  County: SIOUX")
    print("  Horizon: 30 days")
    print("  Forecast points: 3 (simplified)")
    print("  Latency: 1200.5ms")

    # Log a get_historical_sales tool call
    observer.log_tool_execution(
        tool_name="get_historical_sales",
        input_args={"county": "SIOUX", "lookback_days": 60},
        output=[30000.0, 31000.0, 32000.0],  # Simplified output
        latency_ms=500.2,
    )
    print("\n✓ Tool execution logged: get_historical_sales")
    print("  County: SIOUX")
    print("  Lookback: 60 days")
    print("  Historical points: 3 (simplified)")
    print("  Latency: 500.2ms")


def example_4_log_validation_error():
    """Example 4: Log validation errors to LangSmith.

    Demonstrates tracking when user requests violate constraints.
    """
    print("\nExample 4: Log Validation Error")
    print("=" * 60)

    from observability import observer

    # Log a horizon validation error
    observer.log_validation_error(
        thread_id="conv-example-002",
        validation_error="The forecasting model supports up to 30 days ahead. "
        "Please ask for a horizon of 30 days or fewer.",
        resolved_horizon_days=45,
    )
    print("✓ Validation error logged")
    print("  Thread: conv-example-002")
    print("  Error: Horizon exceeds maximum (45 > 30 days)")


def example_5_trace_decorator():
    """Example 5: Use @trace_function decorator.

    Demonstrates the decorator for automatic tracing.
    """
    print("\nExample 5: Trace Decorator")
    print("=" * 60)

    from observability import trace_function
    import time

    @trace_function("compute_forecast_stats")
    def compute_stats(values: list) -> dict:
        """Example function that computes statistics."""
        time.sleep(0.1)  # Simulate some work
        return {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    # Call the traced function
    result = compute_stats([40000, 41000, 42000, 43000])

    print("✓ Function traced with @trace_function decorator")
    print("  Function: compute_forecast_stats")
    print(f"  Result: {result}")
    print("  Check LangSmith for execution trace and latency")


def example_6_lansmith_dashboard():
    """Example 6: Viewing traces in LangSmith dashboard.

    Instructions for accessing and analyzing traces.
    """
    print("\nExample 6: LangSmith Dashboard")
    print("=" * 60)

    from config import settings

    if settings.langsmith_tracing:
        print("✓ LangSmith is enabled")
        print("\nTo view traces:")
        print("1. Open: https://smith.langchain.com/projects")
        print(f"2. Select project: {settings.langsmith_project}")
        print("3. Run the agent and see traces appear in real-time")
        print("\nDashboard features:")
        print("  • Timeline view of graph execution")
        print("  • Individual tool call traces")
        print("  • Performance metrics (latency, throughput)")
        print("  • Error analysis and debugging")
        print("  • Token usage and cost tracking")
    else:
        print("✗ LangSmith is disabled")
        print("  Enable it in .env to use dashboard features")


def example_7_production_setup():
    """Example 7: Production deployment with LangSmith.

    Shows how to configure for Cloud Run or other platforms.
    """
    print("\nExample 7: Production Setup")
    print("=" * 60)

    print("For Cloud Run deployment with LangSmith:")
    print()
    print("1. Create separate .env.prod file:")
    print("   LANGSMITH_TRACING=true")
    print("   LANGSMITH_API_KEY=lsv_your_prod_key")
    print("   LANGSMITH_PROJECT=sales-assistant-agent-prod")
    print()
    print("2. Deploy with environment variables:")
    print("   gcloud run deploy sales-assistant-agent \\")
    print("     --set-env-vars LANGSMITH_TRACING=true \\")
    print("     --update-env-vars LANGSMITH_API_KEY=lsv_... \\")
    print("     --update-env-vars LANGSMITH_PROJECT=sales-assistant-agent-prod")
    print()
    print("3. Separate projects for environments:")
    print("   • sales-assistant-agent-dev (development)")
    print("   • sales-assistant-agent-staging (staging)")
    print("   • sales-assistant-agent-prod (production)")


def run_all_examples():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("LangSmith Observability Examples")
    print("=" * 60)

    example_1_enable_langsmith()
    example_2_log_graph_execution()
    example_3_log_tool_execution()
    example_4_log_validation_error()
    example_5_trace_decorator()
    example_6_lansmith_dashboard()
    example_7_production_setup()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Set LANGSMITH_TRACING=true in .env")
    print("2. Run: poetry run uvicorn main:app --reload")
    print("3. Visit: https://smith.langchain.com/projects")
    print("4. See traces appear in real-time!")


if __name__ == "__main__":
    run_all_examples()
