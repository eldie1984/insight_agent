"""Tests for FastAPI endpoints."""

from unittest.mock import patch


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_returns_ok_status(self, client):
        """Should return OK status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestChatSyncEndpoint:
    """Test /chat/sync endpoint."""

    def test_requires_thread_id_and_message(self, client):
        """Should require both thread_id and message."""
        # Missing thread_id
        response = client.post("/chat/sync", json={"message": "test"})
        assert response.status_code == 422  # Validation error

    def test_returns_chat_sync_response_format(self, client):
        """Should return ChatSyncResponse with reply and optional chart."""
        with patch("main._graph") as mock_graph:
            from langchain_core.messages import AIMessage

            mock_graph.invoke.return_value = {
                "messages": [AIMessage(content="Test response")],
                "chart_data": None,
                "chart_meta": None,
            }

            response = client.post(
                "/chat/sync", json={"thread_id": "test_thread", "message": "Hello"}
            )

            assert response.status_code == 200
            data = response.json()
            assert "reply" in data
            assert "chart" in data

    def test_handles_missing_graph_gracefully(self, client):
        """Should handle missing graph initialization."""
        with patch("main._graph", None):
            response = client.post(
                "/chat/sync", json={"thread_id": "test_thread", "message": "Hello"}
            )

            assert response.status_code == 500


class TestConversationEndpoint:
    """Test /conversation endpoints."""

    def test_get_conversation_returns_history(self, client):
        """Should return conversation history and metadata."""
        # First, make a request to populate memory
        with patch("main._graph") as mock_graph:
            from langchain_core.messages import AIMessage

            mock_graph.invoke.return_value = {
                "messages": [AIMessage(content="Response")],
                "chart_data": None,
                "chart_meta": None,
            }

            client.post(
                "/chat/sync", json={"thread_id": "conv_test", "message": "test message"}
            )

        # Now get the conversation
        response = client.get("/conversation/conv_test")
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "messages" in data
        assert data["summary"]["thread_id"] == "conv_test"

    def test_delete_conversation_clears_history(self, client):
        """Should clear conversation history."""
        with patch("main._graph") as mock_graph:
            from langchain_core.messages import AIMessage

            mock_graph.invoke.return_value = {
                "messages": [AIMessage(content="Response")],
                "chart_data": None,
                "chart_meta": None,
            }

            # Create conversation
            client.post("/chat/sync", json={"thread_id": "del_test", "message": "test"})

            # Delete it
            response = client.delete("/conversation/del_test")
            assert response.status_code == 200
            assert response.json()["status"] == "cleared"

            # Verify it's gone
            response = client.get("/conversation/del_test")
            assert response.status_code == 200
            assert response.json()["summary"]["total_messages"] == 0


class TestChatStreamingEndpoint:
    """Test /chat streaming endpoint."""

    def test_returns_server_sent_events(self, client):
        """Should return SSE format responses."""
        with patch("main._graph") as mock_graph:
            from langchain_core.messages import AIMessage

            mock_graph.invoke.return_value = {
                "messages": [AIMessage(content="Streaming response")],
                "chart_data": None,
                "chart_meta": None,
            }

            response = client.post(
                "/chat", json={"thread_id": "stream_test", "message": "test"}
            )

            assert response.status_code == 200
            # Content-type may include charset (e.g. "text/event-stream; charset=utf-8")
            assert "text/event-stream" in response.headers["content-type"]
            # Should contain event data
            content = response.text
            assert "event:" in content or "data:" in content
