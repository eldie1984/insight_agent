"""Conversation memory management for multi-turn chat."""

import logging
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)


class ConversationMemory:
    """In-memory conversation history storage per thread_id."""

    def __init__(self):
        self.conversations: dict[str, list[BaseMessage]] = {}
        self.created_at: dict[str, datetime] = {}

    def get_messages(self, thread_id: str) -> list[BaseMessage]:
        """Get all messages for a thread."""
        if thread_id not in self.conversations:
            logger.info(f"New conversation thread: {thread_id}")
            self.conversations[thread_id] = []
            self.created_at[thread_id] = datetime.now()
        return self.conversations[thread_id]

    def add_message(self, thread_id: str, message: BaseMessage) -> None:
        """Add a message to the conversation history."""
        if thread_id not in self.conversations:
            self.conversations[thread_id] = []
            self.created_at[thread_id] = datetime.now()

        self.conversations[thread_id].append(message)
        logger.info(
            f"Thread {thread_id}: Added {type(message).__name__} "
            f"(total messages: {len(self.conversations[thread_id])})"
        )

    def clear_thread(self, thread_id: str) -> None:
        """Clear conversation history for a thread."""
        if thread_id in self.conversations:
            del self.conversations[thread_id]
            del self.created_at[thread_id]
            logger.info(f"Cleared conversation thread: {thread_id}")

    def get_summary(self, thread_id: str) -> dict:
        """Get conversation summary."""
        messages = self.get_messages(thread_id)
        human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        ai_count = sum(1 for m in messages if isinstance(m, AIMessage))
        created_ts = None
        if thread_id in self.created_at:
            created_ts = self.created_at[thread_id].isoformat()
        return {
            "thread_id": thread_id,
            "total_messages": len(messages),
            "human_messages": human_count,
            "ai_messages": ai_count,
            "created_at": created_ts,
        }


# Global conversation memory instance
memory = ConversationMemory()
