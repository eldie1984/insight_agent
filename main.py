"""
FastAPI application for the Sales Assistant Agent.

Exposes the /chat (streaming SSE) and /chat/sync (non-streaming) endpoints.
Includes LangSmith observability for tracing and debugging.
"""

from contextlib import asynccontextmanager
import json
import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from graph import build_graph
from config import settings
from observability import observer
from memory import memory


logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    thread_id: str
    message: str


class ChatSyncResponse(BaseModel):
    reply: str
    chart: dict | None = None


# Global graph instance (initialized once at startup)
_graph = None


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Initialize graph on startup."""
    global _graph
    logger.info("Building LangGraph graph...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"LangSmith tracing: {settings.langsmith_tracing}")
    _graph = build_graph()
    logger.info("Graph built successfully")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Sales Assistant Agent",
    description="Conversational agent for sales forecast questions",
    lifespan=lifespan,
)


async def chat_event_generator(thread_id: str, user_message: str):
    """
    Generator that yields SSE events for the /chat endpoint.

    Yields events in the format: "event: <type>\ndata: <json>\n\n"
    Maintains conversation continuity per thread_id.
    """
    from langchain_core.messages import HumanMessage

    try:
        if _graph is None:
            raise RuntimeError("Graph not initialized")

        start_time = time.time()

        # Get conversation history for this thread
        conversation_history = memory.get_messages(thread_id)

        # Create new human message
        user_msg = HumanMessage(content=user_message)
        memory.add_message(thread_id, user_msg)

        # Prepare input state with full conversation history
        input_state = {
            "messages": conversation_history + [user_msg],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": None,
            "validation_error": None,
            "needs_historical_data": False,
        }

        # Initial status
        yield f"event: token\ndata: {json.dumps({'text': 'Processing your request...\\n\\n'})}\n\n"

        messages = input_state.get("messages", [])
        msg_count = len(messages) if isinstance(messages, list) else 0
        logger.info(
            f"Thread {thread_id}: Running graph with {msg_count} total messages"
        )

        # Invoke the graph
        output_state = _graph.invoke(input_state)

        # Extract reply from output
        reply = ""
        ai_response_msg = None

        for message in reversed(output_state.get("messages", [])):
            if hasattr(message, "content") and isinstance(message.content, str):
                reply = message.content
                ai_response_msg = message
                break

        # Store AI response in conversation history
        if ai_response_msg:
            memory.add_message(thread_id, ai_response_msg)
            logger.info(f"Thread {thread_id}: Stored AI response in memory")

        # Stream the reply as tokens (split into words for realistic streaming)
        if reply:
            words = reply.split()
            for i, word in enumerate(words):
                text = word + (" " if i < len(words) - 1 else "")
                yield f"event: token\ndata: {json.dumps({'text': text})}\n\n"

        # Stream chart if available
        if output_state.get("chart_data"):
            chart = {
                "chart_meta": output_state.get("chart_meta", {}),
                "series": [
                    {
                        "date": point["date"],
                        "value": point["value"],
                        "type": point["type"],
                    }
                    for point in output_state["chart_data"]
                ],
            }
            yield f"event: chart\ndata: {json.dumps(chart)}\n\n"

        # Final done event
        latency_ms = (time.time() - start_time) * 1000
        yield f"event: done\ndata: {json.dumps({'latency_ms': round(latency_ms, 0)})}\n\n"

        # Log to observability
        observer.log_graph_execution(
            thread_id=thread_id,
            input_state={"message": user_message},
            output_state={"reply_length": len(reply)},
            latency_ms=latency_ms,
            nodes_executed=["validate_request", "agent", "tools", "chart_builder"],
            chart_generated=output_state.get("chart_data") is not None,
        )

    except Exception as e:
        logger.error(f"Error in chat_event_generator: {e}", exc_info=True)
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"


@app.post("/chat")
async def chat_streaming(request: ChatRequest) -> StreamingResponse:
    """
    Streaming chat endpoint (Server-Sent Events).

    Returns events: token, tool_status, chart, done.
    """
    return StreamingResponse(
        chat_event_generator(request.thread_id, request.message),
        media_type="text/event-stream",
    )


@app.post("/chat/sync")
async def chat_sync(request: ChatRequest) -> ChatSyncResponse:
    """
    Non-streaming chat endpoint (synchronous).

    Returns a single JSON object once the graph run completes.
    Maintains conversation continuity per thread_id.
    """
    from langchain_core.messages import HumanMessage

    start_time = time.time()
    try:
        if _graph is None:
            raise RuntimeError("Graph not initialized")

        # Get conversation history for this thread
        conversation_history = memory.get_messages(request.thread_id)

        # Create new human message
        user_message = HumanMessage(content=request.message)
        memory.add_message(request.thread_id, user_message)

        # Prepare input state for the graph with full conversation history
        input_state = {
            "messages": conversation_history + [user_message],
            "chart_data": None,
            "chart_meta": None,
            "resolved_horizon_days": None,
            "validation_error": None,
            "needs_historical_data": False,
        }

        messages = input_state.get("messages", [])
        msg_count = len(messages) if isinstance(messages, list) else 0
        logger.info(
            f"Thread {request.thread_id}: Running graph with {msg_count} total messages"
        )

        # Invoke the graph
        output_state = _graph.invoke(input_state)

        logger.info(
            f"Output state messages count: {len(output_state.get('messages', []))}"
        )
        logger.info(f"Output state chart_data: {output_state.get('chart_data')}")
        logger.info(f"Output state chart_meta: {output_state.get('chart_meta')}")

        # Extract the reply from the last AI message
        reply = ""
        ai_response_msg = None

        for i, message in enumerate(reversed(output_state.get("messages", []))):
            logger.info(
                f"Message {i}: type={type(message).__name__}, has_content={hasattr(message, 'content')}"
            )
            if hasattr(message, "content"):
                logger.info(f"  content type: {type(message.content).__name__}")
                if isinstance(message.content, str):
                    reply = message.content
                    ai_response_msg = message
                    logger.info(f"  Found reply: {reply[:100]}")
                    break

        if not reply:
            reply = "I couldn't generate a response. Please try again."

        # Store AI response in conversation history
        if ai_response_msg:
            memory.add_message(request.thread_id, ai_response_msg)
            logger.info(f"Thread {request.thread_id}: Stored AI response in memory")

        # Build chart if available
        chart = None
        if output_state.get("chart_data"):
            chart = {
                "chart_meta": output_state.get("chart_meta", {}),
                "series": [
                    {
                        "date": point["date"],
                        "value": point["value"],
                        "type": point["type"],
                    }
                    for point in output_state["chart_data"]
                ],
            }

        latency_ms = (time.time() - start_time) * 1000

        # Log to observability
        observer.log_graph_execution(
            thread_id=request.thread_id,
            input_state={"message": request.message},
            output_state={
                "reply_length": len(reply),
                "chart_generated": chart is not None,
            },
            latency_ms=latency_ms,
            nodes_executed=["validate_request", "agent", "tools", "chart_builder"],
            chart_generated=chart is not None,
        )

        logger.info(
            f"Chat completed in {latency_ms:.0f}ms | "
            f"Thread: {request.thread_id} | "
            f"Chart: {chart is not None}"
        )

        return ChatSyncResponse(reply=reply, chart=chart)

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(f"Error in chat_sync: {e}", exc_info=True)
        observer.log_graph_execution(
            thread_id=request.thread_id,
            input_state={"message": request.message},
            output_state={"error": str(e)},
            latency_ms=latency_ms,
            nodes_executed=[],
            chart_generated=False,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/conversation/{thread_id}")
async def get_conversation(thread_id: str):
    """Get conversation history for a thread."""
    messages = memory.get_messages(thread_id)
    summary = memory.get_summary(thread_id)

    # Convert messages to JSON-serializable format
    messages_json = [
        {
            "type": type(msg).__name__,
            "content": msg.content if hasattr(msg, "content") else None,
        }
        for msg in messages
    ]

    return {"summary": summary, "messages": messages_json}


@app.delete("/conversation/{thread_id}")
async def clear_conversation(thread_id: str):
    """Clear conversation history for a thread."""
    memory.clear_thread(thread_id)
    return {"status": "cleared", "thread_id": thread_id}
