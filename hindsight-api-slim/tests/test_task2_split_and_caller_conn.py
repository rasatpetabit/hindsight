"""Focused tests for task-2 transactional refactor capabilities.

Covers the two capabilities this task introduced beyond the pre-existing behavior:

1. **Dedup adjudication/mutation split** (design §4.4): ``_dedup_reconcile_create`` /
   ``_dedup_reconcile_update`` accept a pre-computed ``outcome`` so Phase A (adjudication) can
   run separately and Phase B (CAS fold) is called with it — no re-adjudication, no LLM call.
2. **Caller-owned connection seam** (design §4.2/§4.5): ``_write_group(pool, conn)`` uses the
   provided ``conn`` without opening a nested transaction, so all observation writes and source
   marks can share one logical write-group in Phase B.

These are deterministic unit tests (no DB, no real LLM) matching the repo's dedup-test style.
"""

import types
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

from hindsight_api.engine.consolidation.consolidator import (
    _dedup_fold_create,
    _dedup_reconcile_create,
    _dedup_reconcile_update,
    _DedupOutcome,
    _write_group,
)
from hindsight_api.engine.memories import CASOutcome


@dataclass
class _FakeObs:
    id: str
    text: str


def _obs(text: str, sim: float, oid: str = "33333333-3333-4333-8333-333333333333"):
    from hindsight_api.engine.search.types import RetrievalResult

    return RetrievalResult(id=oid, text=text, fact_type="observation", similarity=sim)


_TWIN_ID = "33333333-3333-4333-8333-333333333333"
_UPDATED_ID = "44444444-4444-4444-8444-444444444444"


def _snap(unit_id, sources=(), revision="rev-1"):
    return types.SimpleNamespace(
        memory=types.SimpleNamespace(source_memory_ids=list(sources)),
        revision=revision,
    )


class _Backend:
    """Backend-shaped stand-in matching acquire_with_retry's ``_wraps_backend`` path."""

    _wraps_backend = True

    def __init__(self, conn=None):
        self._conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self._conn or _Conn()


class _Conn:
    """Records whether a transaction was opened (to prove caller-owned mode does NOT nest)."""

    def __init__(self):
        self._in_txn = False
        self.transaction_calls = 0
        self.executes = []
        self.fetch = AsyncMock(return_value=[{"id": uuid.uuid4()}])

    @asynccontextmanager
    async def transaction(self):
        self.transaction_calls += 1
        self._in_txn = True
        try:
            yield
        finally:
            self._in_txn = False

    async def execute(self, query, *args):
        self.executes.append((query, args))


def _make_store(fold_outcome=CASOutcome.APPLIED):
    store = types.SimpleNamespace(
        recall_unified=AsyncMock(),
        snapshot_memories=AsyncMock(return_value=[_snap(_TWIN_ID)]),
        cas_fold_observation=AsyncMock(return_value=fold_outcome),
        cas_delete_memory=AsyncMock(return_value=CASOutcome.APPLIED),
        writes_sql=True,
    )
    store.writes_memory_rows_in_sql_for = lambda bank_id: store.writes_sql
    return store


def _patch_store(store):
    from contextlib import ExitStack
    from unittest.mock import patch as _patch

    stack = ExitStack()
    stack.enter_context(_patch("hindsight_api.engine.memories.get_memories", lambda: store))
    stack.enter_context(_patch("hindsight_api.engine.consolidation.consolidator.get_memories", lambda: store))
    return stack


def _base_kwargs(**overrides):
    kwargs = dict(
        pool=types.SimpleNamespace(),
        memory_engine=types.SimpleNamespace(embeddings=object()),
        bank_id="bank1",
        config=types.SimpleNamespace(
            consolidation_dedup_threshold=0.97,
            text_search_extension="native",
            text_search_extension_native_language="english",
        ),
        dedup_llm_config=types.SimpleNamespace(call=AsyncMock()),
        create_text="YouTube content in Uzbek is very rich.",
        create_source_ids=[uuid.uuid4()],
        tags=["t1"],
    )
    kwargs.update(overrides)
    return kwargs


# ── capability 1: adjudication/mutation split ──────────────────────────────


async def test_create_with_precomputed_outcome_skips_adjudication() -> None:
    """Passing ``outcome`` must skip the LLM/adjudication entirely and go straight to fold."""
    kwargs = _base_kwargs(pool=_Backend(), conn=_Conn())
    llm = kwargs["dedup_llm_config"]
    store = _make_store()
    outcome = _DedupOutcome(best_id=_TWIN_ID, merged_text="merged text", should_merge=True)
    with (
        patch("hindsight_api.engine.memories.get_memories", lambda: store),
        patch("hindsight_api.engine.consolidation.consolidator.get_memories", lambda: store),
        patch(
            "hindsight_api.engine.consolidation.consolidator._filter_live_source_memories",
            AsyncMock(return_value=kwargs["create_source_ids"]),
        ),
    ):
        result = await _dedup_reconcile_create(outcome=outcome, expected_revision="phase-a-twin", **kwargs)
    assert result == _TWIN_ID
    llm.call.assert_not_called()  # adjudication skipped — no LLM call
    store.cas_fold_observation.assert_awaited_once()
    assert store.cas_fold_observation.call_args.kwargs["merged_text"] == "merged text"
    assert store.cas_fold_observation.call_args.kwargs["expected_revision"] == "phase-a-twin"


async def test_update_with_precomputed_outcome_skips_adjudication() -> None:
    """UPDATE-path split: pre-computed outcome skips re-adjudication and folds + deletes."""
    store = _make_store()

    async def _snapshots(conn=None, fq_table=None, bank_id=None, unit_ids=None):
        if unit_ids == [_TWIN_ID]:
            return [_snap(_TWIN_ID)]
        return [_snap(_UPDATED_ID)]

    store.snapshot_memories.side_effect = None
    store.snapshot_memories.side_effect = _snapshots
    conn = _Conn()
    llm = types.SimpleNamespace(call=AsyncMock())
    kwargs = dict(
        pool=_Backend(conn),
        memory_engine=types.SimpleNamespace(embeddings=object()),
        bank_id="bank1",
        config=types.SimpleNamespace(
            consolidation_dedup_threshold=0.97,
            text_search_extension="native",
            text_search_extension_native_language="english",
        ),
        dedup_llm_config=llm,
        updated_id=_UPDATED_ID,
        updated_text="Uzbek content on YouTube is very rich and growing.",
        updated_emb_str="[0.1, 0.2, 0.3]",
        tags=["t1"],
    )
    outcome = _DedupOutcome(best_id=_TWIN_ID, merged_text="merged text", should_merge=True)
    with (
        patch("hindsight_api.engine.memories.get_memories", lambda: store),
        patch("hindsight_api.engine.consolidation.consolidator.get_memories", lambda: store),
        patch(
            "hindsight_api.engine.consolidation.consolidator._filter_live_source_memories",
            AsyncMock(return_value=[uuid.uuid4()]),
        ),
    ):
        await _dedup_reconcile_update(outcome=outcome, **kwargs)
    llm.call.assert_not_called()  # adjudication skipped
    store.cas_fold_observation.assert_awaited_once()
    store.cas_delete_memory.assert_awaited_once()


# ── capability 2: caller-owned connection (no nested transaction) ──────────


async def test_write_group_calller_mode_yields_existing_conn_no_nested_txn() -> None:
    """``_write_group(pool, conn)`` with a caller conn must NOT open a transaction on it."""
    conn = _Conn()
    async with _write_group(types.SimpleNamespace(), conn) as c:
        assert c is conn  # yielded the caller's connection
        assert conn.transaction_calls == 0  # no nested transaction opened
    assert conn.transaction_calls == 0


async def test_write_group_serial_mode_opens_transaction() -> None:
    """Serial path (conn=None) must still acquire + open a transaction (unchanged behavior)."""
    conn = _Conn()
    backend = _Backend(conn)
    async with _write_group(backend) as c:
        assert c is conn
        assert c.transaction_calls == 1  # serial path opens exactly one transaction
    assert c.transaction_calls == 1


async def test_dedup_fold_create_calller_mode_no_nested_txn() -> None:
    """The CAS fold on a caller-owned conn must not open a transaction either."""
    store = _make_store()
    conn = _Conn()
    outcome = _DedupOutcome(best_id=_TWIN_ID, merged_text="merged text", should_merge=True)
    live_ids = [uuid.uuid4()]
    with (
        patch("hindsight_api.engine.memories.get_memories", lambda: store),
        patch("hindsight_api.engine.consolidation.consolidator.get_memories", lambda: store),
        patch(
            "hindsight_api.engine.consolidation.consolidator._filter_live_source_memories",
            AsyncMock(return_value=live_ids),
        ),
    ):
        result = await _dedup_fold_create(
            store=store,
            conn=conn,
            memory_engine=types.SimpleNamespace(embeddings=object()),
            bank_id="bank1",
            config=types.SimpleNamespace(),
            outcome=outcome,
            create_source_ids=live_ids,
            txn=None,
            expected_revision="phase-a-twin",
        )
    assert result == _TWIN_ID
    assert conn.transaction_calls == 0  # never opened a nested transaction
