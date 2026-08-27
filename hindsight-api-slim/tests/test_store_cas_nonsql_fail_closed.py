"""Task 4: non-SQL UPDATE/DELETE/fold must use CAS or fail closed.

A store with ``writes_memory_rows_in_sql_for=False`` must never blind-write via
``upsert_observation``, ``delete_facts``, or ``_reconcile_merge_via_store``.
Base CAS defaults raise ``CASNotSupportedError``. A store that implements the
CAS seam is invoked through ``cas_*`` only.
"""

from __future__ import annotations

import types
import uuid
from contextlib import ExitStack, asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.engine.consolidation import consolidator as C
from hindsight_api.engine.memories.base import (
    CASNotSupportedError,
    CASOutcome,
    MemorySnapshot,
    StoredMemory,
)
from hindsight_api.engine.response_models import MemoryFact

OBS_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
TWIN_ID = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
UPD_ID = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"


def _mem(unit_id: str, text: str = "obs") -> StoredMemory:
    return StoredMemory(
        unit_id=unit_id,
        text=text,
        fact_type="observation",
        tags=["t1"],
        proof_count=1,
        source_memory_ids=["s1"],
    )


class _Conn:
    @asynccontextmanager
    async def transaction(self):
        yield self

    async def execute(self, query, *args):
        return None


class _NonSqlStore:
    """Non-SQL store. CAS methods inherit fail-closed behaviour unless overridden."""

    writes_memory_rows_in_sql = False

    def __init__(self, *, cas_impl=None):
        self.cas_impl = cas_impl
        self.cas_update_calls: list[dict] = []
        self.cas_delete_calls: list[dict] = []
        self.cas_fold_calls: list[dict] = []
        self.upsert_calls: list[dict] = []
        self.delete_facts_calls: list[dict] = []
        self.get_memories_calls: list[dict] = []
        self.snapshots = {
            OBS_ID: MemorySnapshot(memory=_mem(OBS_ID), revision="phase-a"),
            TWIN_ID: MemorySnapshot(memory=_mem(TWIN_ID), revision="phase-a-twin"),
            UPD_ID: MemorySnapshot(
                memory=_mem(UPD_ID, "updated"),
                revision="phase-a-upd",
            ),
        }

    def writes_memory_rows_in_sql_for(self, _bank_id: str) -> bool:
        return False

    async def snapshot_memories(self, *, conn, fq_table, bank_id, unit_ids):
        return [self.snapshots[str(u)] for u in unit_ids if str(u) in self.snapshots]

    async def get_memories(self, *, conn, fq_table, bank_id, unit_ids):
        self.get_memories_calls.append({"unit_ids": list(unit_ids)})
        return [self.snapshots[str(u)].memory for u in unit_ids if str(u) in self.snapshots]

    async def upsert_observation(self, **kwargs):
        self.upsert_calls.append(kwargs)

    async def delete_facts(self, bank_id, unit_ids, *, txn=None):
        self.delete_facts_calls.append({"bank_id": bank_id, "unit_ids": list(unit_ids), "txn": txn})

    async def cas_update_memory(self, **kwargs):
        self.cas_update_calls.append(kwargs)
        if self.cas_impl is None:
            raise CASNotSupportedError("base CAS default")
        return await self.cas_impl.cas_update_memory(**kwargs)

    async def cas_delete_memory(self, **kwargs):
        self.cas_delete_calls.append(kwargs)
        if self.cas_impl is None:
            raise CASNotSupportedError("base CAS default")
        return await self.cas_impl.cas_delete_memory(**kwargs)

    async def cas_fold_observation(self, **kwargs):
        self.cas_fold_calls.append(kwargs)
        if self.cas_impl is None:
            raise CASNotSupportedError("base CAS default")
        return await self.cas_impl.cas_fold_observation(**kwargs)


class _WorkingCAS:
    async def cas_update_memory(self, **kwargs):
        return CASOutcome.APPLIED

    async def cas_delete_memory(self, **kwargs):
        return CASOutcome.APPLIED

    async def cas_fold_observation(self, **kwargs):
        return CASOutcome.APPLIED


def _engine():
    return types.SimpleNamespace(
        embeddings=object(),
        _backend=types.SimpleNamespace(ops=types.SimpleNamespace(uses_observation_sources_table=False)),
    )


def _config():
    return types.SimpleNamespace(
        enable_observation_history=False,
        text_search_extension="native",
        text_search_extension_native_language="english",
        consolidation_dedup_threshold=0.97,
    )


def _patches(store, reconcile_spy):
    src = uuid.uuid4()
    patches = [
        patch.object(C, "get_memories", return_value=store),
        patch.object(C, "get_config", return_value=_config()),
        patch.object(C, "_filter_live_source_memories", new=AsyncMock(return_value=[src])),
        patch.object(C, "_native_search_vector_update", return_value=None),
        patch.object(C, "_append_observation_history", new=AsyncMock()),
        patch.object(C, "_delete_observation_history", new=AsyncMock()),
        patch.object(C, "record_created_memory_ids", lambda *_a, **_k: None),
    ]
    if hasattr(C, "_reconcile_merge_via_store"):
        patches.append(patch.object(C, "_reconcile_merge_via_store", new=reconcile_spy))
    return patches, src


async def _run_update(store, reconcile_spy):
    patches, src = _patches(store, reconcile_spy)
    fact = MemoryFact(id=OBS_ID, text="obs", fact_type="observation", tags=["t1"])
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        await C._execute_update_action(
            pool=object(),
            memory_engine=_engine(),
            bank_id="b",
            source_memory_ids=[src],
            observation_id=OBS_ID,
            new_text="rewritten",
            observations=[fact],
            conn=_Conn(),
            precomputed_embedding="[0.1]",
            expected_revision="phase-a",
        )


async def _run_delete(store, reconcile_spy):
    patches, _src = _patches(store, reconcile_spy)
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        await C._execute_delete_action(
            conn=_Conn(),
            bank_id="b",
            observation_id=OBS_ID,
            expected_revision="phase-a",
        )


async def _run_fold_create(store, reconcile_spy, *, merged_embedding=None):
    patches, src = _patches(store, reconcile_spy)
    outcome = C._DedupOutcome(
        best_id=TWIN_ID,
        merged_text="merged",
        should_merge=True,
        candidate_ids={TWIN_ID},
        candidate_revisions={TWIN_ID: "phase-a-twin"},
        merged_embedding=merged_embedding,
    )
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        await C._dedup_fold_create(
            store=store,
            conn=_Conn(),
            memory_engine=_engine(),
            bank_id="b",
            config=_config(),
            outcome=outcome,
            create_source_ids=[src],
            expected_revision="phase-a-twin",
        )


async def _run_fold_update(store, reconcile_spy, *, merged_embedding=None):
    patches, _src = _patches(store, reconcile_spy)
    outcome = C._DedupOutcome(
        best_id=TWIN_ID,
        merged_text="merged",
        should_merge=True,
        candidate_ids={TWIN_ID},
        candidate_revisions={TWIN_ID: "phase-a-twin"},
        merged_embedding=merged_embedding,
    )
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        await C._dedup_fold_update(
            store=store,
            conn=_Conn(),
            memory_engine=_engine(),
            bank_id="b",
            config=_config(),
            outcome=outcome,
            updated_id=UPD_ID,
            updated_text="updated",
        )


def _assert_no_bypasses(store, reconcile_spy):
    assert store.upsert_calls == [], f"upsert_observation called: {store.upsert_calls}"
    assert store.delete_facts_calls == [], f"delete_facts called: {store.delete_facts_calls}"
    assert reconcile_spy.await_count == 0, f"_reconcile_merge_via_store called {reconcile_spy.await_count}x"


# --------------------------------------------------------------------------- negative: base CAS defaults fail closed


@pytest.mark.asyncio
async def test_nonsql_update_raises_cas_not_supported_and_skips_upsert():
    store = _NonSqlStore()
    reconcile = AsyncMock()
    with pytest.raises(CASNotSupportedError):
        await _run_update(store, reconcile)
    _assert_no_bypasses(store, reconcile)
    assert store.cas_update_calls, "cas_update_memory must be invoked"


@pytest.mark.asyncio
async def test_nonsql_delete_raises_cas_not_supported_and_skips_delete_facts():
    store = _NonSqlStore()
    reconcile = AsyncMock()
    with pytest.raises(CASNotSupportedError):
        await _run_delete(store, reconcile)
    _assert_no_bypasses(store, reconcile)
    assert store.cas_delete_calls, "cas_delete_memory must be invoked"


@pytest.mark.asyncio
async def test_nonsql_fold_create_raises_cas_not_supported_and_skips_reconcile():
    store = _NonSqlStore()
    reconcile = AsyncMock()
    with pytest.raises(CASNotSupportedError):
        await _run_fold_create(store, reconcile)
    _assert_no_bypasses(store, reconcile)
    assert store.cas_fold_calls, "cas_fold_observation must be invoked"


@pytest.mark.asyncio
async def test_nonsql_fold_update_raises_cas_not_supported_and_skips_reconcile():
    store = _NonSqlStore()
    reconcile = AsyncMock()
    with pytest.raises(CASNotSupportedError):
        await _run_fold_update(store, reconcile)
    _assert_no_bypasses(store, reconcile)
    assert store.cas_fold_calls, "cas_fold_observation must be invoked"


# --------------------------------------------------------------------------- positive: working CAS is used, bypasses stay at zero


@pytest.mark.asyncio
async def test_nonsql_update_calls_cas_update_not_upsert():
    store = _NonSqlStore(cas_impl=_WorkingCAS())
    reconcile = AsyncMock()
    await _run_update(store, reconcile)
    _assert_no_bypasses(store, reconcile)
    assert len(store.cas_update_calls) == 1
    assert store.cas_update_calls[0]["expected_revision"] == "phase-a"
    assert store.cas_update_calls[0]["unit_id"] == OBS_ID


@pytest.mark.asyncio
async def test_nonsql_delete_calls_cas_delete_not_delete_facts():
    store = _NonSqlStore(cas_impl=_WorkingCAS())
    reconcile = AsyncMock()
    await _run_delete(store, reconcile)
    _assert_no_bypasses(store, reconcile)
    assert len(store.cas_delete_calls) == 1
    assert store.cas_delete_calls[0]["expected_revision"] == "phase-a"
    assert store.cas_delete_calls[0]["unit_id"] == OBS_ID


@pytest.mark.asyncio
async def test_fold_create_passes_merged_embedding():
    store = _NonSqlStore(cas_impl=_WorkingCAS())
    reconcile = AsyncMock()
    await _run_fold_create(store, reconcile, merged_embedding="[0.77]")
    assert store.cas_fold_calls, "cas_fold_observation was not called"
    assert store.cas_fold_calls[0].get("merged_embedding") == "[0.77]"
    assert store.cas_fold_calls[0]["merged_text"] == "merged"


@pytest.mark.asyncio
async def test_fold_update_passes_merged_embedding():
    store = _NonSqlStore(cas_impl=_WorkingCAS())
    reconcile = AsyncMock()
    await _run_fold_update(store, reconcile, merged_embedding="[0.88]")
    assert store.cas_fold_calls, "cas_fold_observation was not called"
    assert store.cas_fold_calls[0].get("merged_embedding") == "[0.88]"
    assert store.cas_fold_calls[0]["merged_text"] == "merged"


@pytest.mark.asyncio
async def test_nonsql_fold_create_calls_cas_fold_not_reconcile():
    store = _NonSqlStore(cas_impl=_WorkingCAS())
    reconcile = AsyncMock()
    await _run_fold_create(store, reconcile)
    _assert_no_bypasses(store, reconcile)
    assert len(store.cas_fold_calls) == 1
    assert store.cas_fold_calls[0]["expected_revision"] == "phase-a-twin"
    assert store.cas_fold_calls[0]["observation_id"] == TWIN_ID


@pytest.mark.asyncio
async def test_nonsql_fold_update_calls_cas_fold_and_delete_not_reconcile():
    store = _NonSqlStore(cas_impl=_WorkingCAS())
    reconcile = AsyncMock()
    await _run_fold_update(store, reconcile)
    _assert_no_bypasses(store, reconcile)
    assert len(store.cas_fold_calls) == 1
    assert store.cas_fold_calls[0]["observation_id"] == TWIN_ID
    assert len(store.cas_delete_calls) == 1
    assert store.cas_delete_calls[0]["unit_id"] == UPD_ID


class _FoldAppliedDeleteOutcome:
    def __init__(self, delete_outcome: CASOutcome):
        self.delete_outcome = delete_outcome

    async def cas_update_memory(self, **kwargs):
        return CASOutcome.APPLIED

    async def cas_fold_observation(self, **kwargs):
        return CASOutcome.APPLIED

    async def cas_delete_memory(self, **kwargs):
        return self.delete_outcome


@pytest.mark.parametrize("delete_outcome", [CASOutcome.STALE, CASOutcome.MISSING])
@pytest.mark.asyncio
async def test_fold_update_non_applied_delete_raises_batch_stale(delete_outcome, caplog):
    """Task 6: a STALE/MISSING leftover-row delete must abort the batch.

    History must not be deleted and the success log must not fire — the fold already
    applied, so rolling back the whole transaction is the only safe fate.
    """
    store = _NonSqlStore(cas_impl=_FoldAppliedDeleteOutcome(delete_outcome))
    history = AsyncMock()
    reconcile = AsyncMock()
    patches, _src = _patches(store, reconcile)
    # Replace the default history stub with a spy we can assert on.
    patches = [p for p in patches if getattr(p, "attribute", None) != "_delete_observation_history"]
    patches.append(patch.object(C, "_delete_observation_history", new=history))
    outcome = C._DedupOutcome(
        best_id=TWIN_ID,
        merged_text="merged",
        should_merge=True,
        candidate_ids={TWIN_ID},
        candidate_revisions={TWIN_ID: "phase-a-twin"},
    )
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        with pytest.raises(C._BatchStaleError) as exc:
            await C._dedup_fold_update(
                store=store,
                conn=_Conn(),
                memory_engine=_engine(),
                bank_id="b",
                config=_config(),
                outcome=outcome,
                updated_id=UPD_ID,
                updated_text="updated",
            )
    assert UPD_ID in str(exc.value)
    history.assert_not_awaited()
    assert "dedup-merged updated observation" not in caplog.text
