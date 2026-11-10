"""Read-only SQLite connections kept apart from the one that writes.

The chunk store and the file registry share ``knowledge.sqlite3`` in WAL mode, where a
reader sees the last committed state and never waits for a write. Reading on the write
connection loses that: under its lock, a search waits for a whole indexing transaction,
and a registry lookup waits behind an update that is itself waiting for the store's
write lock. A small pool of read connections, opened as reads overlap, keeps them apart.
"""

from __future__ import annotations

from contextlib import contextmanager
import sqlite3
import threading
from typing import Callable, Iterator


class ReaderPool:
    def __init__(self, connect: Callable[[], sqlite3.Connection], size: int = 2):
        self._connect = connect
        self.size = max(1, int(size))
        self._idle: list[sqlite3.Connection] = []
        self._opened = 0
        self._closed = False
        self._cond = threading.Condition()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """A connection for this thread until the block ends; waits when all are in use."""
        conn = self._take()
        try:
            yield conn
        finally:
            self._give(conn)

    def _take(self) -> sqlite3.Connection:
        with self._cond:
            while not self._idle and self._opened >= self.size and not self._closed:
                self._cond.wait()
            if self._closed:
                raise sqlite3.ProgrammingError("Cannot operate on a closed database.")
            if self._idle:
                # The most recently used one: its page cache is the warmest.
                return self._idle.pop()
            self._opened += 1
        try:
            return self._connect()
        except BaseException:
            with self._cond:
                self._opened -= 1
                self._cond.notify()
            raise

    def _give(self, conn: sqlite3.Connection) -> None:
        with self._cond:
            if self._closed:
                self._opened -= 1
                conn.close()
            else:
                self._idle.append(conn)
            self._cond.notify()

    def close(self) -> None:
        """Close the idle connections now and the busy ones as they are returned."""
        with self._cond:
            self._closed = True
            idle, self._idle = self._idle, []
            self._opened -= len(idle)
            self._cond.notify_all()
        for conn in idle:
            conn.close()
