"""
Q-Ravens Checkpoint Module

Provides state persistence using SQLite for checkpointing.
Enables pause/resume of workflows across restarts.
"""

import os
from pathlib import Path
from typing import Optional

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.memory import MemorySaver

from q_ravens.core.config import settings


def get_checkpoint_dir() -> Path:
    """Get the checkpoint directory, creating it if needed."""
    checkpoint_dir = Path(settings.chroma_persist_dir).parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def get_checkpoint_path(session_id: Optional[str] = None) -> str:
    """Get the path for the checkpoint database."""
    checkpoint_dir = get_checkpoint_dir()
    if session_id:
        return str(checkpoint_dir / f"session_{session_id}.db")
    return str(checkpoint_dir / "q_ravens.db")


async def create_sqlite_checkpointer(session_id: Optional[str] = None) -> AsyncSqliteSaver:
    """
    Create an async SQLite checkpointer for state persistence.

    Args:
        session_id: Optional session ID for isolated checkpoints

    Returns:
        AsyncSqliteSaver instance
    """
    db_path = get_checkpoint_path(session_id)
    return AsyncSqliteSaver.from_conn_string(db_path)


def create_memory_checkpointer() -> MemorySaver:
    """
    Create an in-memory checkpointer (no persistence).

    Useful for testing or when persistence is not needed.

    Returns:
        MemorySaver instance
    """
    return MemorySaver()


class CheckpointManager:
    """
    Manager for handling checkpoints across sessions.

    Provides methods to:
    - Create new sessions
    - Resume existing sessions
    - List available sessions
    - Clean up old sessions
    """

    def __init__(self, persist: bool = True):
        """
        Initialize the checkpoint manager.

        Args:
            persist: Whether to persist checkpoints to disk
        """
        self.persist = persist
        self.checkpoint_dir = get_checkpoint_dir() if persist else None

    def list_sessions(self) -> list[dict]:
        """
        List all available checkpoint sessions.

        Returns:
            List of session info dictionaries
        """
        if not self.persist or not self.checkpoint_dir:
            return []

        sessions = []
        for db_file in self.checkpoint_dir.glob("session_*.db"):
            session_id = db_file.stem.replace("session_", "")
            stat = db_file.stat()
            sessions.append({
                "session_id": session_id,
                "path": str(db_file),
                "size_kb": stat.st_size / 1024,
                "modified": stat.st_mtime,
            })

        return sorted(sessions, key=lambda x: x["modified"], reverse=True)

    def session_exists(self, session_id: str) -> bool:
        """Check if a session checkpoint exists."""
        if not self.persist:
            return False
        path = Path(get_checkpoint_path(session_id))
        return path.exists()

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session checkpoint.

        Args:
            session_id: The session ID to delete

        Returns:
            True if deleted, False if not found
        """
        if not self.persist:
            return False

        path = Path(get_checkpoint_path(session_id))
        if path.exists():
            path.unlink()
            return True
        return False

    def cleanup_old_sessions(self, max_sessions: int = 10) -> int:
        """
        Clean up old sessions, keeping only the most recent ones.

        Args:
            max_sessions: Maximum number of sessions to keep

        Returns:
            Number of sessions deleted
        """
        sessions = self.list_sessions()
        if len(sessions) <= max_sessions:
            return 0

        deleted = 0
        for session in sessions[max_sessions:]:
            if self.delete_session(session["session_id"]):
                deleted += 1

        return deleted

    async def get_checkpointer(self, session_id: Optional[str] = None):
        """
        Get a checkpointer for the given session.

        Args:
            session_id: Optional session ID

        Returns:
            Checkpointer instance (SQLite or Memory)
        """
        if self.persist:
            return await create_sqlite_checkpointer(session_id)
        return create_memory_checkpointer()
