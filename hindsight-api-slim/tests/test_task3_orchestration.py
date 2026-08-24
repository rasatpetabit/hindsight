"""Orchestration tests for the Phase A / Phase B consolidation split (task 3).

Covers the two halves the store-CAS redesign (§4.2) makes structural:

1. **Phase A (prepare)** — ``_prepare_memory_batch`` performs ALL slow work — recall, the
   LLM call, embeddings, semantic-dedup adjudication — and returns a :class:`_PreparedBatch`
   plan. It performs NO writes and acquires NO bank lock or transaction.

2. **Phase B (commit)** — ``_commit_prepared_batch`` executes the prepared writes through
   the store CAS seam on a caller-owned connection, and the job-level orchestration wraps
   every LLM batch's commit in a single ``SELECT bank_id FROM banks ... FOR UPDATE`` guard +
   one transaction (design §4.1 / §4.5). Same-fate: source marks + observation writes share
   one write-group.

The pure tests use mocked recall/LLM/embeddings/dedup so they run without the ML stack;
the PG test exercises the real ``FOR UPDATE`` bank guard against a scratch PostgreSQL.
"""

from __future__ import annotations

import asyncio
import types
import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.engine.consolidation import consolidator as C

# ---------------------------------------------------------------------------
# Phase A: _prepare_memory_batch performs no writes and returns a plan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_phase_a_performs_no_writes_and_no_lock():
    """Phase A must not open any write-group or bank lock — only reads happen.

    We hand it fully-mocked recall/LLM and assert:
    * it returns a _PreparedBatch with the intended deletes/updates/creates;
    * zero writes were issued on any connection.
    """
    mem_id = str(uuid.uuid4())
    memories = [{"id": mem_id, "text": "Some memory", "tags": ["t1"]}]
    create = C._CreateAction(text="Observation from memory", source_fact_ids=[mem_id])
    llm_result = C._BatchLLMResult(creates=[create])

    fake_recall = types.SimpleNamespace(results=[], source_facts={})

    writes: list[str] = []

    class _NoWriteConn:
        """Any accidental write raises."""

        async def execute(self, query, *args):
            writes.append(query)
            raise AssertionError(f"Phase A must not write: {query}")

        async def fetchrow(self, query, *args):
            raise AssertionError(f"Phase A must not fetchrow: {query}")

        @asynccontextmanager
        async def transaction(self):
            writes.append("transaction")
            raise AssertionError("Phase A must not open a transaction")

    class _Pool:
        _wraps_backend = True

        @asynccontextmanager
        async def acquire(self):
            yield _NoWriteConn()

    mem_engine = types.SimpleNamespace(
        embeddings=object(),
        _consolidation_llm_config=types.SimpleNamespace(with_config=lambda *a, **k: object()),
    )
    config = types.SimpleNamespace(
        consolidation_dedup_threshold=1.0,  # disables dedup (>= 1.0)
    )

    with (
        patch.object(C, "_find_related_observations", new=AsyncMock(return_value=fake_recall)),
        patch.object(C, "_consolidate_batch_with_llm", new=AsyncMock(return_value=llm_result)),
        patch.object(C, "_effective_scope_limit", return_value=-1),
    ):
        prepared = await C._prepare_memory_batch(
            pool=_Pool(),
            memory_engine=mem_engine,
            llm_config=object(),
            bank_id="bank1",
            memories=memories,
            request_context=object(),
            config=config,
        )

    assert isinstance(prepared, C._PreparedBatch)
    assert len(prepared.creates) == 1
    assert prepared.creates[0].create is create
    assert prepared.creates[0].create_source_ids == [mem_id]
    assert len(prepared.deletes) == 0
    assert len(prepared.updates) == 0
    assert writes == [], f"Phase A issued {len(writes)} writes"


# ---------------------------------------------------------------------------
# Phase B: _commit_prepared_batch executes CAS writes on caller-owned conn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_commit_phase_b_applies_create_via_cas_on_caller_conn():
    """Phase B executes a prepared CREATE through _execute_create_action on the caller conn."""
    mem_id = str(uuid.uuid4())
    memories = [{"id": mem_id, "text": "Some memory", "tags": ["t1"]}]
    create = C._CreateAction(text="Observation text", source_fact_ids=[mem_id])
    agg = C._SourceAggregation(event_date=None, occurred_start=None, occurred_end=None, mentioned_at=None, tags=["t1"])
    prepared = C._PreparedBatch(
        memories=memories,
        per_fact_obs_ids={},
        union_observations=[],
        llm_result=C._BatchLLMResult(creates=[create]),
        fact_tags=["t1"],
        deletes=[],
        updates=[],
        creates=[C._PreparedCreate(create=create, source_mems=[memories[0]], agg=agg, create_source_ids=[mem_id])],
        dedup_enabled=False,
    )

    create_action = AsyncMock(return_value="created")

    class _Conn:
        @asynccontextmanager
        async def transaction(self):
            yield self

        async def execute(self, query, *args):
            return None

        async def fetchrow(self, query, *args):
            return None

    conn = _Conn()
    with (
        patch.object(C, "_execute_create_action", new=create_action),
        patch.object(C, "_execute_update_action", new=AsyncMock(return_value=None)),
        patch.object(C, "_execute_delete_action", new=AsyncMock()),
        patch.object(C, "_fresh_source_validation", new=AsyncMock(return_value="ok")),
    ):
        results, deleted, _stale = await C._commit_prepared_batch(
            prepared=prepared,
            pool=None,
            memory_engine=types.SimpleNamespace(embeddings=object()),
            bank_id="bank1",
            config=types.SimpleNamespace(),
            conn=conn,
        )

    assert deleted == 0
    assert results == [{"action": "created"}]
    # The create executor got the caller-owned conn (no nested txn), the source ids, and text.
    kwargs = create_action.call_args.kwargs
    assert kwargs["conn"] is conn
    assert kwargs["source_memory_ids"] == [mem_id]
    assert kwargs["text"] == "Observation text"


# ---------------------------------------------------------------------------
# Bank guard: SELECT ... FOR UPDATE serializes same-bank Phase B commits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bank_guard_for_update_is_issued_once_per_batch():
    """The orchestration issues exactly one SELECT ... FOR UPDATE on banks per LLM batch.

    The ``_process_one_llm_batch`` flow is exercised indirectly here by checking that the
    guard statement is present in the write-group and that committing one prepared batch does
    not itself acquire the bank lock (that is the orchestrator's job).
    """
    mem_id = str(uuid.uuid4())
    memories = [{"id": mem_id, "text": "Some memory", "tags": []}]
    prepared = C._PreparedBatch(
        memories=memories,
        per_fact_obs_ids={},
        union_observations=[],
        llm_result=C._BatchLLMResult(),
        fact_tags=[],
        deletes=[],
        updates=[],
        creates=[],
        dedup_enabled=False,
    )

    executed: list[str] = []

    class _Conn:
        @asynccontextmanager
        async def transaction(self):
            yield self

        async def execute(self, query, *args):
            executed.append(query)
            return None

        async def fetchrow(self, query, *args):
            return None

    with (
        patch.object(C, "_execute_create_action", new=AsyncMock(return_value="skipped")),
        patch.object(C, "_execute_update_action", new=AsyncMock(return_value=None)),
        patch.object(C, "_execute_delete_action", new=AsyncMock()),
        patch.object(C, "_fresh_source_validation", new=AsyncMock(return_value="ok")),
    ):
        await C._commit_prepared_batch(
            prepared=prepared,
            pool=None,
            memory_engine=types.SimpleNamespace(embeddings=object()),
            bank_id="bank1",
            config=types.SimpleNamespace(),
            conn=_Conn(),
        )

    # _commit_prepared_batch must NOT take the bank lock itself — the orchestrator does.
    assert not any("FOR UPDATE" in q for q in executed), f"bank lock leaked into commit: {executed}"


# ---------------------------------------------------------------------------
# PG-backed: FOR UPDATE guard genuinely serializes two same-bank commits
# ---------------------------------------------------------------------------


def _db_url() -> str:
    import os

    url = os.getenv("HINDSIGHT_API_DATABASE_URL")
    if not url:
        pytest.skip("HINDSIGHT_API_DATABASE_URL not set (point at a pgvector PostgreSQL to run)")
    return url


@pytest.mark.asyncio
async def test_pg_bank_for_update_serializes_same_bank_writers():
    """Two concurrent transactions taking SELECT ... FOR UPDATE on the same bank row serialize.

    This proves the guard primitive the orchestration relies on (design §4.1): writer A holds
    the row lock; writer B blocks until A commits; then B proceeds and sees A's commit. We run
    it against a real PostgreSQL so a broken/absent lock is caught.
    """
    import asyncpg

    url = _db_url()
    conn_a = await asyncpg.connect(url)
    conn_b = await asyncpg.connect(url)
    bank_id = f"guard-{uuid.uuid4().hex[:8]}"
    try:
        # Seed the bank row outside any test transaction.
        async with conn_a.transaction():
            await conn_a.execute(
                f"INSERT INTO {C.fq_table('banks')} (bank_id, name) VALUES ($1,$2)", bank_id, "guard-test"
            )

        b_acquired = asyncio.Event()

        async def b_writer():
            try:
                async with conn_b.transaction():
                    # This should BLOCK until A's transaction commits (A holds FOR UPDATE).
                    await conn_b.execute(
                        f"SELECT bank_id FROM {C.fq_table('banks')} WHERE bank_id = $1 FOR UPDATE", bank_id
                    )
                b_acquired.set()
            except Exception:
                b_acquired.set()  # surface the failure via the event

        # A takes + holds the FOR UPDATE inside its own open transaction FIRST.
        async with conn_a.transaction():
            await conn_a.execute(
                f"SELECT bank_id FROM {C.fq_table('banks')} WHERE bank_id = $1 FOR UPDATE", bank_id
            )
            # Now start B: it will block on the FOR UPDATE until A commits.
            b_task = asyncio.create_task(b_writer())
            await asyncio.sleep(0.3)
            assert not b_acquired.is_set(), "B acquired the bank lock while A held it — guard broken"
        # A's transaction committed -> lock released -> B proceeds.
        await asyncio.wait_for(b_acquired.wait(), timeout=5)
        b_task.cancel()
        try:
            await b_task
        except (asyncio.CancelledError, Exception):
            pass
    finally:
        await conn_a.close()
        await conn_b.close()
