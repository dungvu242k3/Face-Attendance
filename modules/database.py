"""Database connection management.

Repositories use :class:`Database.transaction` so every request gets its own
connection from a thread-safe pool and transactions are always committed or
rolled back in one place.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from psycopg2.pool import ThreadedConnectionPool

from config import Settings


class Database:
    """Small transaction boundary around a psycopg2 connection pool."""

    def __init__(self, settings: Settings):
        max_connections = max(settings.db_min_connections, settings.db_max_connections)
        self._pool = ThreadedConnectionPool(
            minconn=settings.db_min_connections,
            maxconn=max_connections,
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            host=settings.db_host,
            port=settings.db_port,
            sslmode=settings.db_sslmode,
            **(
                {"sslrootcert": settings.db_sslrootcert}
                if settings.db_sslrootcert
                else {}
            ),
        )

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        connection = self._pool.getconn()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            self._pool.putconn(connection, close=bool(connection.closed))

    def close(self) -> None:
        self._pool.closeall()
