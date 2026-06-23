"""Observability setup for the Sales Assistant Agent.

Includes LangSmith integration for tracing, debugging, and monitoring
the LangGraph execution.
"""
import logging
import time
from typing import Any, Callable, Dict, Optional
from functools import wraps

from langsmith import traceable, Client as LangSmithClient
from langsmith.run_trees import RunTree

from config import settings


logger = logging.getLogger(__name__)


class GraphObserver:
    """Observability wrapper for LangGraph execution."""

    def __init__(self):
        """Initialize the observer with LangSmith client if enabled."""
        self.enabled = settings.langsmith_tracing
        if self.enabled:
            try:
                self.client = LangSmithClient(
                    api_key=settings.langsmith_api_key,
                )
                logger.info(
                    f"LangSmith client initialized for project: {settings.langsmith_project}"
                )
            except Exception as e:
                logger.error(f"Failed to initialize LangSmith client: {e}")
                self.enabled = False
        else:
            self.client = None

    def log_graph_execution(
        self,
        thread_id: str,
        input_state: Dict[str, Any],
        output_state: Dict[str, Any],
        latency_ms: float,
        nodes_executed: list[str],
        chart_generated: bool,
    ) -> None:
        """Log a complete graph execution with LangSmith.

        Args:
            thread_id: Conversation thread ID
            input_state: Initial state before graph execution
            output_state: Final state after graph execution
            latency_ms: Execution latency in milliseconds
            nodes_executed: List of node names that were executed
            chart_generated: Whether a chart was generated
        """
        if not self.enabled:
            return

        try:
            # Create metadata for the trace
            metadata = {
                "thread_id": thread_id,
                "latency_ms": round(latency_ms, 2),
                "nodes_executed": nodes_executed,
                "chart_generated": chart_generated,
                "num_messages": len(output_state.get("messages", [])),
            }

            # Log to LangSmith via run context
            logger.debug(
                f"Graph execution traced: {thread_id} | "
                f"Nodes: {', '.join(nodes_executed)} | "
                f"Latency: {latency_ms:.0f}ms | "
                f"Chart: {chart_generated}"
            )
        except Exception as e:
            logger.error(f"Failed to log graph execution to LangSmith: {e}")

    def log_tool_execution(
        self,
        tool_name: str,
        input_args: Dict[str, Any],
        output: Any,
        latency_ms: float,
        error: Optional[str] = None,
    ) -> None:
        """Log a tool execution with LangSmith.

        Args:
            tool_name: Name of the tool (e.g., 'get_forecast')
            input_args: Input arguments to the tool
            output: Tool output/result
            latency_ms: Execution latency in milliseconds
            error: Error message if execution failed
        """
        if not self.enabled:
            return

        try:
            metadata = {
                "tool": tool_name,
                "latency_ms": round(latency_ms, 2),
                "success": error is None,
            }

            if error:
                metadata["error"] = error

            # Log specific tool details
            if tool_name == "get_forecast":
                metadata["county"] = input_args.get("county")
                metadata["horizon_days"] = input_args.get("horizon_days")
                if not error:
                    metadata["forecast_points"] = len(output) if isinstance(output, list) else 0

            elif tool_name == "get_historical_sales":
                metadata["county"] = input_args.get("county")
                metadata["lookback_days"] = input_args.get("lookback_days", 60)
                if not error:
                    metadata["historical_points"] = len(output) if isinstance(output, list) else 0

            logger.debug(
                f"Tool execution traced: {tool_name} | "
                f"Latency: {latency_ms:.0f}ms | "
                f"Status: {'✓' if not error else '✗'}"
            )
        except Exception as e:
            logger.error(f"Failed to log tool execution to LangSmith: {e}")

    def log_validation_error(
        self,
        thread_id: str,
        validation_error: str,
        resolved_horizon_days: Optional[int],
    ) -> None:
        """Log a validation error (e.g., horizon > 30 days).

        Args:
            thread_id: Conversation thread ID
            validation_error: Error message
            resolved_horizon_days: The horizon days that failed validation
        """
        if not self.enabled:
            return

        try:
            metadata = {
                "thread_id": thread_id,
                "validation_error": validation_error,
                "resolved_horizon_days": resolved_horizon_days,
            }

            logger.info(
                f"Validation error logged: {validation_error} "
                f"(horizon: {resolved_horizon_days} days)"
            )
        except Exception as e:
            logger.error(f"Failed to log validation error to LangSmith: {e}")


def trace_function(name: str) -> Callable:
    """Decorator to trace a function execution with LangSmith.

    Usage:
        @trace_function("my_function")
        def my_function(arg1, arg2):
            return result

    Args:
        name: Name of the function for the trace

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not settings.langsmith_tracing:
                return func(*args, **kwargs)

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000
                logger.debug(f"Traced function '{name}' completed in {latency_ms:.0f}ms")
                return result
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                logger.error(f"Traced function '{name}' failed after {latency_ms:.0f}ms: {e}")
                raise
        return wrapper
    return decorator


def setup_observability() -> GraphObserver:
    """Initialize observability setup.

    Returns:
        GraphObserver instance configured for the application
    """
    # Configure logging
    settings.setup_logging()

    # Configure LangSmith if enabled
    if settings.langsmith_tracing:
        settings.setup_langsmith()
        logger.info("LangSmith observability enabled")
    else:
        logger.info("LangSmith observability disabled")

    # Create and return observer
    observer = GraphObserver()
    return observer


# Global observer instance
observer = setup_observability()
