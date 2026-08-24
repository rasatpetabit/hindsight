"""Durable SQLite-backed MemoriesExtension for the authentic M5 integration seam.

This is a **persistent, transaction-aware** fixture (NOT a dict fake). It keeps
observation rows, source facts, consolidation marks, and cross-store write-group
transactions in a real SQLite file so a "crash" (dropping the object and
reopening the DB) actually loses no durable state and recovery must reconcile it.

Cross-store txn semantics (matching upstream ``memories/base.py``):

- ``mint_txn`` creates a pending write-group handle with no DB transaction open.
- A writer tags its writes with that handle; they land in a ``pending_writes``
  table and are NOT visible through ``get_memories``/``scan_memories`` until the
  group is decided committed.
- ``write_txn_witness`` records a witness row (inside the caller's txn, here a
  SQLite transaction over the same file).
- ``decide_txn(commit=True)`` publishes every pending write of the group and
  removes its pending rows; ``decide_txn(commit=False)`` discards them.
- ``recover_pending_txns`` is the real backstop algorithm: for each bank, a
  pending group whose witness row exists is COMMITTED; one with no witness past
  ``grace_seconds`` is ABORTED; one with no witness still within grace is left
  pending. Witness rows older than ``witness_ttl_seconds`` are reaped.

This store reports ``writes_memory_rows_in_sql_for -> False`` (it owns its rows
outside Postgres), which is exactly the store class the upstream maintenance loop
runs cross-store txn recovery for (``MaintenanceLoop._run_txn_recovery`` skips
SQL stores).

The store implements the narrow `MemoriesExtension` surface exercised by the
transformed ``_commit_one_llm_batch`` / ``_commit_memory_batch`` path so a real
write-group can run end-to-end against durable rows.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Reuse the host shim's FactRecord shape (extended by this module's own fields).
try:
    from .memories import FactRecord
except Exception:  # pragma: no cover - fallback for standalone import
    from dataclasses import dataclass, field

    @dataclass
    class FactRecord:
        id: str = ""
        text: str = ""
        tags: list = field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class _Row(dict):
    """dict with attribute access, mirroring asyncpg Row shape used upstream."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


class DurableMemoryStore:
    """Persistent non-SQL memories store with cross-store write-group txns."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._lock = asyncio.Lock()
        self._init_db()

    # ------------------------------------------------------------------ schema

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_units (
                    unit_id      TEXT PRIMARY KEY,
                    bank_id      TEXT NOT NULL,
                    text         TEXT,
                    fact_type    TEXT,
                    embedding    TEXT,
                    proof_count  INTEGER DEFAULT 0,
                    source_memory_ids TEXT,
                    tags         TEXT,
                    event_date   TEXT,
                    occurred_start TEXT,
                    occurred_end TEXT,
                    mentioned_at TEXT,
                    created_at   TEXT,
                    updated_at   TEXT
                );
                CREATE TABLE IF NOT EXISTS consolidation_marks (
                    unit_id TEXT NOT NULL,
                    bank_id TEXT NOT NULL,
                    failed  INTEGER NOT NULL DEFAULT 0,
                    when_iso TEXT,
                    PRIMARY KEY (unit_id, bank_id)
                );
                CREATE TABLE IF NOT EXISTS pending_txns (
                    txn_id   TEXT PRIMARY KEY,
                    bank_id  TEXT NOT NULL,
                    mutating INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    state    TEXT NOT NULL DEFAULT 'pending' -- pending|committed|aborted
                );
                CREATE TABLE IF NOT EXISTS txn_witnesses (
                    txn_id   TEXT PRIMARY KEY,
                    bank_id  TEXT NOT NULL,
                    witnessed_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pending_writes (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    txn_id     TEXT NOT NULL,
                    unit_id    TEXT NOT NULL,
                    bank_id    TEXT NOT NULL,
                    op         TEXT NOT NULL, -- create|update|delete|mark_success|mark_failed
                    payload    TEXT NOT NULL   -- JSON record/mark data
                );
                """
            )

    # ----------------------------------------------------------- txn lifecycle

    async def mint_txn(self, *, bank_id: str, mutating: bool) -> dict:
        async with self._lock:
            txn_id = str(uuid.uuid4())
            with self._conn() as c:
                c.execute(
                    "INSERT INTO pending_txns (txn_id, bank_id, mutating, created_at, state)"
                    " VALUES (?, ?, ?, ?, 'pending')",
                    (txn_id, bank_id, 1 if mutating else 0, asyncio.get_event_loop().time()),
                )
            return {"id": txn_id, "bank_id": bank_id}

    async def write_txn_witness(self, txn: dict | None, *, conn=None, fq_table=None) -> None:
        if txn is None or not txn.get("id"):
            return
        async with self._lock:
            with self._conn() as c:
                c.execute(
                    "INSERT OR REPLACE INTO txn_witnesses (txn_id, bank_id, witnessed_at)"
                    " VALUES (?, ?, ?)",
                    (txn["id"], txn.get("bank_id", ""), asyncio.get_event_loop().time()),
                )

    async def decide_txn(self, txn: dict | None, *, commit: bool) -> None:
        if txn is None or not txn.get("id"):
            return
        txn_id = txn["id"]
        async with self._lock:
            with self._conn() as c:
                row = c.execute(
                    "SELECT state FROM pending_txns WHERE txn_id = ?", (txn_id,)
                ).fetchone()
                if row is None or row["state"] != "pending":
                    return  # already decided; idempotent no-op
                if commit:
                    # Publish every pending write of this group into durable rows.
                    writes = c.execute(
                        "SELECT * FROM pending_writes WHERE txn_id = ?", (txn_id,)
                    ).fetchall()
                    for w in writes:
                        payload = json.loads(w["payload"])
                        op = w["op"]
                        if op in ("create", "update"):
                            self._apply_upsert(c, w["unit_id"], w["bank_id"], payload)
                        elif op == "delete":
                            c.execute(
                                "DELETE FROM memory_units WHERE unit_id = ? AND bank_id = ?",
                                (w["unit_id"], w["bank_id"]),
                            )
                        elif op == "mark_success":
                            c.execute(
                                "INSERT OR REPLACE INTO consolidation_marks"
                                " (unit_id, bank_id, failed, when_iso) VALUES (?, ?, 0, ?)",
                                (w["unit_id"], w["bank_id"], payload.get("when")),
                            )
                        elif op == "mark_failed":
                            c.execute(
                                "INSERT OR REPLACE INTO consolidation_marks"
                                " (unit_id, bank_id, failed, when_iso) VALUES (?, ?, 1, ?)",
                                (w["unit_id"], w["bank_id"], payload.get("when")),
                            )
                    c.execute(
                        "DELETE FROM pending_writes WHERE txn_id = ?", (txn_id,)
                    )
                else:
                    # Abort: discard held writes (they were never visible).
                    c.execute("DELETE FROM pending_writes WHERE txn_id = ?", (txn_id,))
                c.execute(
                    "UPDATE pending_txns SET state = ? WHERE txn_id = ?",
                    ("committed" if commit else "aborted", txn_id),
                )

    async def recover_pending_txns(
        self,
        *,
        conn=None,
        fq_table=None,
        bank_ids: list[str],
        first_seen: dict[str, float],
        now: float,
        grace_seconds: float = 300.0,
        witness_ttl_seconds: float = 3600.0,
    ) -> int:
        """Real backstop: decide each bank's undecided groups against witnesses."""
        decided = 0
        async with self._lock:
            with self._conn() as c:
                for bank in bank_ids:
                    pending = c.execute(
                        "SELECT * FROM pending_txns WHERE bank_id = ? AND state = 'pending'",
                        (bank,),
                    ).fetchall()
                    for p in pending:
                        txn_id = p["txn_id"]
                        first = first_seen.get(txn_id, now)
                        witness = c.execute(
                            "SELECT witnessed_at FROM txn_witnesses WHERE txn_id = ?",
                            (txn_id,),
                        ).fetchone()
                        if witness is not None:
                            # Witness present => the writer committed; publish.
                            writes = c.execute(
                                "SELECT * FROM pending_writes WHERE txn_id = ?", (txn_id,)
                            ).fetchall()
                            for w in writes:
                                payload = json.loads(w["payload"])
                                op = w["op"]
                                if op in ("create", "update"):
                                    self._apply_upsert(c, w["unit_id"], w["bank_id"], payload)
                                elif op == "delete":
                                    c.execute(
                                        "DELETE FROM memory_units WHERE unit_id = ? AND bank_id = ?",
                                        (w["unit_id"], w["bank_id"]),
                                    )
                                elif op == "mark_success":
                                    c.execute(
                                        "INSERT OR REPLACE INTO consolidation_marks"
                                        " (unit_id, bank_id, failed, when_iso) VALUES (?, ?, 0, ?)",
                                        (w["unit_id"], w["bank_id"], payload.get("when")),
                                    )
                                elif op == "mark_failed":
                                    c.execute(
                                        "INSERT OR REPLACE INTO consolidation_marks"
                                        " (unit_id, bank_id, failed, when_iso) VALUES (?, ?, 1, ?)",
                                        (w["unit_id"], w["bank_id"], payload.get("when")),
                                    )
                            c.execute("DELETE FROM pending_writes WHERE txn_id = ?", (txn_id,))
                            c.execute(
                                "UPDATE pending_txns SET state = 'committed' WHERE txn_id = ?",
                                (txn_id,),
                            )
                            decided += 1
                        elif now - first >= grace_seconds:
                            # No witness past grace => writer aborted/crashed before commit.
                            c.execute("DELETE FROM pending_writes WHERE txn_id = ?", (txn_id,))
                            c.execute(
                                "UPDATE pending_txns SET state = 'aborted' WHERE txn_id = ?",
                                (txn_id,),
                            )
                            decided += 1
                        # else: still within grace — leave pending for a later tick.
                # Reap expired witnesses.
                cutoff = now - witness_ttl_seconds
                c.execute("DELETE FROM txn_witnesses WHERE witnessed_at < ?", (cutoff,))
        return decided

    # ------------------------------------------------------------- store surface

    def writes_memory_rows_in_sql_for(self, bank_id: str) -> bool:
        return False

    def _apply_upsert(self, c: sqlite3.Connection, unit_id: str, bank_id: str, rec: dict) -> None:
        rec.setdefault("fact_type", "observation")
        rec.setdefault("proof_count", len(rec.get("source_memory_ids") or []))
        rec.setdefault("tags", [])
        rec.setdefault("created_at", _now_iso())
        rec.setdefault("updated_at", _now_iso())
        c.execute(
            """
            INSERT INTO memory_units (
                unit_id, bank_id, text, fact_type, embedding, proof_count,
                source_memory_ids, tags, event_date, occurred_start,
                occurred_end, mentioned_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(unit_id) DO UPDATE SET
                text=excluded.text,
                fact_type=excluded.fact_type,
                embedding=excluded.embedding,
                proof_count=excluded.proof_count,
                source_memory_ids=excluded.source_memory_ids,
                tags=excluded.tags,
                event_date=excluded.event_date,
                occurred_start=excluded.occurred_start,
                occurred_end=excluded.occurred_end,
                mentioned_at=excluded.mentioned_at,
                updated_at=excluded.updated_at
            """,
            (
                unit_id,
                bank_id,
                rec.get("text"),
                rec.get("fact_type"),
                rec.get("embedding"),
                rec.get("proof_count", 0),
                json.dumps(rec.get("source_memory_ids") or []),
                json.dumps(rec.get("tags") or []),
                rec.get("event_date"),
                rec.get("occurred_start"),
                rec.get("occurred_end"),
                rec.get("mentioned_at"),
                rec.get("created_at"),
                rec.get("updated_at"),
            ),
        )

    async def upsert_observation(self, *, conn=None, bank_id: str, txn=None, record) -> None:
        rec_data = self._record_to_dict(record)
        if txn is not None and txn.get("id"):
            # Held write: invisible until the group is decided committed.
            async with self._lock:
                with self._conn() as c:
                    op = (
                        "create"
                        if not self._unit_exists(c, str(record.unit_id), bank_id)
                        else "update"
                    )
                    unit_exists = self._unit_exists(c, str(record.unit_id), bank_id)
                    c.execute(
                        "INSERT INTO pending_writes (txn_id, unit_id, bank_id, op, payload)"
                        " VALUES (?, ?, ?, ?, ?)",
                        (
                            txn["id"],
                            str(record.unit_id),
                            bank_id,
                            op if unit_exists else ("update" if False else op),
                            json.dumps(rec_data),
                        ),
                    )
            return
        async with self._lock:
            with self._conn() as c:
                self._apply_upsert(c, str(record.unit_id), bank_id, rec_data)

    def _unit_exists(self, c: sqlite3.Connection, unit_id: str, bank_id: str) -> bool:
        return (
            c.execute(
                "SELECT 1 FROM memory_units WHERE unit_id = ? AND bank_id = ?",
                (unit_id, bank_id),
            ).fetchone()
            is not None
        )

    @staticmethod
    def _record_to_dict(record) -> dict:
        d: dict[str, Any] = {}
        for field in (
            "unit_id", "id", "text", "fact_type", "embedding", "proof_count",
            "source_memory_ids", "tags", "event_date", "occurred_start",
            "occurred_end", "mentioned_at", "created_at", "updated_at",
        ):
            v = getattr(record, field, None)
            if v is not None and field not in d:
                # Durable serialization: datetimes become ISO strings.
                if hasattr(v, "isoformat"):
                    v = v.isoformat()
                elif isinstance(v, (list, tuple)):
                    v = [x.isoformat() if hasattr(x, "isoformat") else x for x in v]
                d[field] = v
        # FactRecord uses `id` as its primary key in the shim; upsert callers pass `unit_id`.
        if d.get("id") and not d.get("unit_id"):
            d["unit_id"] = d.pop("id")
        return d

    async def delete_facts(self, bank_ids, unit_ids, *, txn=None) -> None:
        """Remove observation rows (durable). Honors the write-group when a txn is given.

        The transformed call site invokes ``delete_facts(bank_id, [observation_id], txn=txn)``
        where ``bank_id`` is a str and ``unit_ids`` a list. Normalize defensively.
        """
        banks = [bank_ids] if isinstance(bank_ids, str) else list(bank_ids or [])
        uid_list = []
        for u in unit_ids or []:
            if isinstance(u, (list, tuple)):
                uid_list.extend(u)
            else:
                uid_list.append(u)
        async with self._lock:
            with self._conn() as c:
                for bid in banks or [""]:
                    for uid in uid_list:
                        if txn is not None and txn.get("id"):
                            c.execute(
                                "INSERT INTO pending_writes (txn_id, unit_id, bank_id, op, payload)"
                                " VALUES (?, ?, ?, 'delete', ?)",
                                (txn["id"], str(uid), bid, "{}"),
                            )
                        else:
                            c.execute(
                                "DELETE FROM memory_units WHERE unit_id = ? AND bank_id = ?",
                                (str(uid), bid),
                            )

    async def get_memories(self, *, conn=None, fq_table=None, bank_id=None,
                           unit_ids=None) -> list[_Row]:
        """Return durable observation rows. Pending writes are NEVER visible here."""
        async with self._lock:
            with self._conn() as c:
                rows = []
                if unit_ids is None:
                    cur = c.execute(
                        "SELECT * FROM memory_units WHERE bank_id = ? ORDER BY updated_at",
                        (bank_id or "",),
                    )
                else:
                    placeholders = ",".join("?" * len(unit_ids))
                    cur = c.execute(
                        f"SELECT * FROM memory_units WHERE unit_id IN ({placeholders})",
                        list(unit_ids),
                    )
                for r in cur.fetchall():
                    rows.append(self._row_from_sqlite(r))
                return rows

    @staticmethod
    def _row_from_sqlite(r: sqlite3.Row) -> _Row:
        out: dict[str, Any] = {}
        for k in r.keys():
            out[k] = r[k]
        # Upstream Memory rows expose BOTH ``unit_id`` and ``id``; callers use both.
        if "unit_id" in out and "id" not in out:
            out["id"] = out["unit_id"]
        out.setdefault("source_memory_ids", [])
        try:
            smi = out.get("source_memory_ids")
            out["source_memory_ids"] = (
                json.loads(smi) if isinstance(smi, str) else smi or []
            )
        except Exception:
            out["source_memory_ids"] = []
        try:
            t = out.get("tags")
            out["tags"] = json.loads(t) if isinstance(t, str) else t or []
        except Exception:
            out["tags"] = []
        return _Row(out)

    async def scan_memories(self, *a, **k) -> list[_Row]:
        """Non-locking memory scan for existence checks."""
        return await self.get_memories(**k)

    async def count_unconsolidated(self) -> int:
        async with self._lock:
            with self._conn() as c:
                return int(c.execute(
                    """
                    SELECT count(*) FROM memory_units mu
                    LEFT JOIN consolidation_marks cm ON mu.unit_id=cm.unit_id AND mu.bank_id=cm.bank_id
                    WHERE cm.unit_id IS NULL OR cm.failed != 0 OR cm.when_iso IS NULL OR mu.fact_type='fact'
                    """
                ).fetchone()[0])

    async def find_unconsolidated(self) -> list[_Row]:
        async with self._lock:
            with self._conn() as c:
                cur = c.execute(
                    """
                    SELECT mu.* FROM memory_units mu
                    LEFT JOIN consolidation_marks cm ON mu.unit_id=cm.unit_id AND mu.bank_id=cm.bank_id
                    WHERE cm.unit_id IS NULL OR cm.failed != 0 OR cm.when_iso IS NULL OR mu.fact_type='fact'
                    ORDER BY mu.updated_at LIMIT 200
                    """
                )
                return [self._row_from_sqlite(r) for r in cur.fetchall()]

    async def recall_unified(self):
        return []

    async def mark_consolidated(self, *, conn=None, fq_table=None, bank_id=None,
                                unit_ids=None, when=None, failed=False, txn=None):
        """Tag consolidation marks; deferred via the write-group when a txn is supplied."""
        assert unit_ids is not None
        mark_payloads = []
        for uid in unit_ids:
            mark_payloads.append({"when": when.isoformat() if hasattr(when, "isoformat") else when})
        if txn is not None and txn.get("id"):
            async with self._lock:
                with self._conn() as c:
                    for uid in unit_ids:
                        c.execute(
                            "INSERT INTO pending_writes (txn_id, unit_id, bank_id, op, payload)"
                            " VALUES (?, ?, ?, ?, ?)",
                            (
                                txn["id"],
                                str(uid),
                                bank_id or "",
                                "mark_failed" if failed else "mark_success",
                                json.dumps({"when": mark_payloads[0].get("when")}),
                            ),
                        )
            return
        async with self._lock:
            with self._conn() as c:
                for uid in unit_ids:
                    c.execute(
                        "INSERT OR REPLACE INTO consolidation_marks"
                        " (unit_id, bank_id, failed, when_iso) VALUES (?, ?, ?, ?)",
                        (
                            str(uid),
                            bank_id or "",
                            1 if failed else 0,
                            mark_payloads[0].get("when") or _now_iso(),
                        ),
                    )

    # ------------------------------------------------------------- test helpers

    def marks(self) -> dict[str, bool]:
        """Durable map: unit-id -> True (consolidated-ok) / False (failed)."""
        out: dict[str, bool] = {}
        with self._conn() as c:
            for r in c.execute("SELECT unit_id, failed FROM consolidation_marks").fetchall():
                out[r["unit_id"]] = not bool(r["failed"])
        return out

    def pending_txn_states(self) -> dict[str, str]:
        """Durable map: txn-id -> state (pending|committed|aborted)."""
        out: dict[str, str] = {}
        with self._conn() as c:
            for r in c.execute("SELECT txn_id, state FROM pending_txns").fetchall():
                out[r["txn_id"]] = r["state"]
        return out

    def pending_write_count(self) -> int:
        """Number of held writes not yet published/aborted."""
        with self._conn() as c:
            return int(c.execute("SELECT count(*) FROM pending_writes").fetchone()[0])

    def witness_count(self) -> int:
        with self._conn() as c:
            return int(c.execute("SELECT count(*) FROM txn_witnesses").fetchone()[0])
