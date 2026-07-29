from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class SupportDatabase:
    def __init__(self, path: str | Path = "data/support.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS customers (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL,
                    plan TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active'
                );
                CREATE TABLE IF NOT EXISTS tickets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT,
                    subject TEXT NOT NULL, details TEXT NOT NULL, priority TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT NOT NULL,
                    answer TEXT NOT NULL, rating INTEGER NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT,
                    question TEXT NOT NULL, answer TEXT NOT NULL, intent TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
            """)
            db.execute("INSERT OR IGNORE INTO customers(id,name,email,plan,status) VALUES(?,?,?,?,?)", ("C-1001", "Jordan Lee", "jordan@example.com", "Pro", "active"))
            db.execute("INSERT OR IGNORE INTO customers(id,name,email,plan,status) VALUES(?,?,?,?,?)", ("C-1002", "Sam Rivera", "sam@example.com", "Starter", "past_due"))

    def customer(self, customer_id: str | None) -> dict[str, str] | None:
        if not customer_id:
            return None
        with self.connection() as db:
            row = db.execute("SELECT id,name,email,plan,status FROM customers WHERE id=?", (customer_id,)).fetchone()
        return dict(row) if row else None

    def create_ticket(self, subject: str, details: str, *, customer_id: str | None = None, priority: str = "normal") -> int:
        with self.connection() as db:
            cursor = db.execute("INSERT INTO tickets(customer_id,subject,details,priority) VALUES(?,?,?,?)", (customer_id, subject, details, priority))
            return int(cursor.lastrowid)

    def tickets(self, limit: int = 50) -> list[dict[str, object]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self.connection() as db:
            rows = db.execute("SELECT * FROM tickets ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def add_feedback(self, question: str, answer: str, rating: int) -> None:
        if rating not in {-1, 1}:
            raise ValueError("rating must be -1 or 1")
        with self.connection() as db:
            db.execute("INSERT INTO feedback(question,answer,rating) VALUES(?,?,?)", (question, answer, rating))

    def record_interaction(self, question: str, answer: str, intent: str, customer_id: str | None = None) -> None:
        with self.connection() as db:
            db.execute(
                "INSERT INTO interactions(customer_id,question,answer,intent) VALUES(?,?,?,?)",
                (customer_id, question, answer, intent),
            )

    def interactions(self, customer_id: str | None = None, limit: int = 20) -> list[dict[str, object]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self.connection() as db:
            if customer_id:
                rows = db.execute("SELECT * FROM interactions WHERE customer_id=? ORDER BY id DESC LIMIT ?", (customer_id, limit)).fetchall()
            else:
                rows = db.execute("SELECT * FROM interactions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]
