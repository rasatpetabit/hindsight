"""Task 2: Phase-A target/candidate tokens must be compared under the bank guard.

These tests pin the store-CAS redesign (judge 76f0ac68 / remediation plan Task 2):

1. Prepared UPDATE/DELETE/fold objects carry the token of the StoredMemory the
   LLM/adjudication actually consumed — not a later re-snapshot.
2. ``_prevalidate_prepared_batch`` stales on revision mismatch AND missing row
   for UPDATE targets, DELETE targets, fold twins, every decision-relevant
   candidate, and every union observation shown to the main LLM.
3. After a matching prevalidate, mutation executors pass that Phase-A token to
   ``cas_*`` as ``expected_revision``. An executor-side fresh snapshot offering a
   different token must not be used.
"""

from __future__ import annotations

import types
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.engine.consolidation import consolidator as C
from hindsight_api.engine.memories.base import (
    CASOutcome,
    MemorySnapshot,
    StoredMemory,
    memory_revision_token,
)
from hindsight_api.engine.response_models import MemoryFact


def _mem(**overrides) -> StoredMemory:
    defaults = dict(
        unit_id="u1",
        text="Ada designed the first algorithm.",
        fact_type="observation",
        context="history",
        tags=["programming", "history"],
        metadata={"source": "notes"},
        proof_count=2,
        source_memory_ids=["s1", "s2"],
        event_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
        occurred_start=datetime(2024, 1, 14, tzinfo=timezone.utc),
        occurred_end=datetime(2024, 1, 16, tzinfo=timezone.utc),
        mentioned_at=datetime(2024, 1, 17, tzinfo=timezone.utc),
        created_at=datetime(2024, 1, 10, tzinfo=timezone.utc),
        observation_scopes=["harness:pi"],
        consolidated_at=None,
    )
    defaults.update(overrides)
    return StoredMemory(**defaults)


def _agg() -> C._SourceAggregation:
    return C._SourceAggregation(event_date=None, occurred_start=None, occurred_end=None, mentioned_at=None, tags=["t1"])


def _empty_batch(**overrides) -> C._PreparedBatch:
    kwargs = dict(
        memories=[],
        per_fact_obs_ids={},
        union_observations=[],
        llm_result=C._BatchLLMResult(),
        fact_tags=["t1"],
        deletes=[],
        updates=[],
        creates=[],
        dedup_enabled=False,
        observation_revisions={},
    )
    kwargs.update(overrides)
    return C._PreparedBatch(**kwargs)


class _FakeStore:
    """In-memory snapshot/CAS spy used by prevalidate and executor tests."""

    def __init__(self, snapshots: dict[str, MemorySnapshot | None]):
        self.snapshots = snapshots
        self.cas_update_calls: list[dict] = []
        self.cas_delete_calls: list[dict] = []
        self.cas_fold_calls: list[dict] = []

    def writes_memory_rows_in_sql_for(self, _bank_id: str) -> bool:
        return True

    async def snapshot_memories(self, *, conn, fq_table, bank_id, unit_ids):
        out: list[MemorySnapshot] = []
        for uid in unit_ids:
            snap = self.snapshots.get(str(uid), None)
            if snap is None:
                continue
            out.append(snap)
        return out

    async def cas_update_memory(self, **kwargs):
        self.cas_update_calls.append(kwargs)
        return CASOutcome.APPLIED

    async def cas_delete_memory(self, **kwargs):
        self.cas_delete_calls.append(kwargs)
        return CASOutcome.APPLIED

    async def cas_fold_observation(self, **kwargs):
        self.cas_fold_calls.append(kwargs)
        return CASOutcome.APPLIED


class _Conn:
    @asynccontextmanager
    async def transaction(self):
        yield self

    async def execute(self, query, *args):
        return None

    async def fetchrow(self, query, *args):
        return None

    async def fetchval(self, query, *args):
        return 1


def _obs_fact(unit_id: str, text: str) -> MemoryFact:
    return MemoryFact(id=unit_id, text=text, fact_type="observation", tags=["t1"])


# --------------------------------------------------------------------------- prepared-object fields


def test_prepared_update_carries_phase_a_revision():
    fields = getattr(C._PreparedUpdate, "__dataclass_fields__")
    assert "phase_a_revision" in fields


def test_prepared_delete_carries_phase_a_revision():
    assert hasattr(C, "_PreparedDelete")
    fields = getattr(C._PreparedDelete, "__dataclass_fields__")
    assert "phase_a_revision" in fields


def test_prepared_create_carries_fold_target_and_candidate_revisions():
    fields = getattr(C._PreparedCreate, "__dataclass_fields__")
    assert "phase_a_target_revision" in fields
    assert "candidate_revisions" in fields


def test_prepared_batch_carries_observation_revisions():
    fields = getattr(C._PreparedBatch, "__dataclass_fields__")
    assert "observation_revisions" in fields


def test_dedup_outcome_carries_candidate_revisions():
    fields = getattr(C._DedupOutcome, "__dataclass_fields__")
    assert "candidate_revisions" in fields


# --------------------------------------------------------------------------- prevalidate mismatch / missing


@pytest.mark.asyncio
async def test_prevalidate_stales_on_update_revision_mismatch():
    consumed = _mem(unit_id="obs-upd", text="phase-a text")
    later = _mem(unit_id="obs-upd", text="mutated during llm window")
    store = _FakeStore({"obs-upd": MemorySnapshot(memory=later, revision=memory_revision_token(later))})
    src = str(uuid.uuid4())
    pupd = C._PreparedUpdate(
        update=C._UpdateAction(text="new", observation_id="obs-upd", source_fact_ids=[src]),
        source_mems=[{"id": src, "text": "src"}],
        agg=_agg(),
        embedding_str=None,
        phase_a_revision=memory_revision_token(consumed),
    )
    prepared = _empty_batch(
        updates=[pupd],
        memories=[{"id": src, "text": "src"}],
        observation_revisions={"obs-upd": memory_revision_token(consumed)},
        source_snapshots={},
    )
    with (
        patch.object(C, "get_memories", return_value=store),
        patch.object(C, "_fresh_source_validation", new=AsyncMock(return_value="ok")),
        patch.object(C, "_observation_exists", new=AsyncMock(return_value=True)),
    ):
        reason = await C._prevalidate_prepared_batch(prepared, conn=_Conn(), bank_id="b")
    assert reason.startswith("update_target_stale:") or "obs-upd" in reason
    assert reason != "ok"


@pytest.mark.asyncio
async def test_prevalidate_stales_on_update_target_missing():
    consumed = _mem(unit_id="obs-upd", text="phase-a text")
    store = _FakeStore({})  # missing
    src = str(uuid.uuid4())
    pupd = C._PreparedUpdate(
        update=C._UpdateAction(text="new", observation_id="obs-upd", source_fact_ids=[src]),
        source_mems=[{"id": src, "text": "src"}],
        agg=_agg(),
        embedding_str=None,
        phase_a_revision=memory_revision_token(consumed),
    )
    prepared = _empty_batch(
        updates=[pupd],
        observation_revisions={"obs-upd": memory_revision_token(consumed)},
    )
    with (
        patch.object(C, "get_memories", return_value=store),
        patch.object(C, "_fresh_source_validation", new=AsyncMock(return_value="ok")),
        patch.object(C, "_observation_exists", new=AsyncMock(return_value=False)),
    ):
        reason = await C._prevalidate_prepared_batch(prepared, conn=_Conn(), bank_id="b")
    assert reason != "ok"
    assert "obs-upd" in reason


@pytest.mark.asyncio
async def test_prevalidate_stales_on_delete_revision_mismatch():
    consumed = _mem(unit_id="obs-del", text="phase-a text")
    later = _mem(unit_id="obs-del", text="mutated")
    store = _FakeStore({"obs-del": MemorySnapshot(memory=later, revision=memory_revision_token(later))})
    pdel = C._PreparedDelete(
        delete=C._DeleteAction(observation_id="obs-del", reason="stale"),
        phase_a_revision=memory_revision_token(consumed),
    )
    prepared = _empty_batch(
        deletes=[pdel],
        observation_revisions={"obs-del": memory_revision_token(consumed)},
    )
    with (
        patch.object(C, "get_memories", return_value=store),
        patch.object(C, "_observation_exists", new=AsyncMock(return_value=True)),
    ):
        reason = await C._prevalidate_prepared_batch(prepared, conn=_Conn(), bank_id="b")
    assert reason != "ok"
    assert "obs-del" in reason


@pytest.mark.asyncio
async def test_prevalidate_stales_on_fold_target_mismatch():
    twin = _mem(unit_id="twin", text="phase-a twin")
    later = _mem(unit_id="twin", text="mutated twin")
    store = _FakeStore({"twin": MemorySnapshot(memory=later, revision=memory_revision_token(later))})
    src = uuid.uuid4()
    pcreate = C._PreparedCreate(
        create=C._CreateAction(text="x", source_fact_ids=[str(src)]),
        source_mems=[{"id": src, "text": "src"}],
        agg=_agg(),
        create_source_ids=[src],
        phase_a_target_revision=memory_revision_token(twin),
        candidate_revisions={"twin": memory_revision_token(twin)},
        candidate_ids={"twin"},
        dedup_outcome=C._DedupOutcome(
            best_id="twin",
            merged_text="merged",
            should_merge=True,
            candidate_ids={"twin"},
            candidate_revisions={"twin": memory_revision_token(twin)},
        ),
    )
    prepared = _empty_batch(creates=[pcreate], memories=[{"id": src, "text": "src"}])
    with (
        patch.object(C, "get_memories", return_value=store),
        patch.object(C, "_fresh_source_validation", new=AsyncMock(return_value="ok")),
        patch.object(C, "_semantic_candidate_expansion", new=AsyncMock(return_value="ok")),
    ):
        reason = await C._prevalidate_prepared_batch(prepared, conn=_Conn(), bank_id="b")
    assert reason != "ok"
    assert "twin" in reason


@pytest.mark.asyncio
async def test_prevalidate_stales_on_candidate_revision_mismatch():
    cand = _mem(unit_id="cand", text="phase-a candidate")
    later = _mem(unit_id="cand", text="mutated candidate")
    store = _FakeStore({"cand": MemorySnapshot(memory=later, revision=memory_revision_token(later))})
    src = uuid.uuid4()
    pcreate = C._PreparedCreate(
        create=C._CreateAction(text="x", source_fact_ids=[str(src)]),
        source_mems=[{"id": src, "text": "src"}],
        agg=_agg(),
        create_source_ids=[src],
        phase_a_target_revision=None,
        candidate_revisions={"cand": memory_revision_token(cand)},
        candidate_ids={"cand"},
        dedup_outcome=C._DedupOutcome(
            best_id=None,
            merged_text="",
            should_merge=False,
            candidate_ids={"cand"},
            candidate_revisions={"cand": memory_revision_token(cand)},
        ),
    )
    prepared = _empty_batch(creates=[pcreate])
    with (
        patch.object(C, "get_memories", return_value=store),
        patch.object(C, "_fresh_source_validation", new=AsyncMock(return_value="ok")),
        patch.object(C, "_semantic_candidate_expansion", new=AsyncMock(return_value="ok")),
    ):
        reason = await C._prevalidate_prepared_batch(prepared, conn=_Conn(), bank_id="b")
    assert reason != "ok"
    assert "cand" in reason


@pytest.mark.asyncio
async def test_prevalidate_stales_on_union_observation_mismatch():
    shown = _mem(unit_id="shown", text="shown to llm")
    later = _mem(unit_id="shown", text="mutated after recall")
    store = _FakeStore({"shown": MemorySnapshot(memory=later, revision=memory_revision_token(later))})
    prepared = _empty_batch(
        union_observations=[_obs_fact("shown", shown.text)],
        observation_revisions={"shown": memory_revision_token(shown)},
    )
    with (
        patch.object(C, "get_memories", return_value=store),
        patch.object(C, "_fresh_source_validation", new=AsyncMock(return_value="ok")),
    ):
        reason = await C._prevalidate_prepared_batch(prepared, conn=_Conn(), bank_id="b")
    assert reason != "ok"
    assert "shown" in reason


# --------------------------------------------------------------------------- executor uses Phase-A token, not a fresh snapshot token


@pytest.mark.asyncio
async def test_execute_update_passes_phase_a_revision_not_fresh_snapshot():
    consumed = _mem(unit_id="obs-upd", text="phase-a text")
    later = _mem(unit_id="obs-upd", text="fresh under lock")
    phase_a = memory_revision_token(consumed)
    fresh = memory_revision_token(later)
    assert phase_a != fresh
    store = _FakeStore({"obs-upd": MemorySnapshot(memory=later, revision=fresh)})

    src = uuid.uuid4()
    fact = _obs_fact("obs-upd", consumed.text)
    with (
        patch.object(C, "get_memories", return_value=store),
        patch.object(
            C,
            "get_config",
            return_value=types.SimpleNamespace(
                enable_observation_history=False,
                text_search_extension="native",
                text_search_extension_native_language="english",
            ),
        ),
        patch.object(C, "_filter_live_source_memories", new=AsyncMock(return_value=[src])),
        patch.object(C, "_native_search_vector_update", return_value=None),
        patch.object(C, "_append_observation_history", new=AsyncMock()),
        patch.object(C, "record_created_memory_ids", lambda *_a, **_k: None),
    ):
        await C._execute_update_action(
            pool=object(),
            memory_engine=types.SimpleNamespace(
                _backend=types.SimpleNamespace(ops=types.SimpleNamespace(uses_observation_sources_table=False))
            ),
            bank_id="b",
            source_memory_ids=[src],
            observation_id="obs-upd",
            new_text="rewritten",
            observations=[fact],
            conn=_Conn(),
            precomputed_embedding="[0.1]",
            expected_revision=phase_a,
        )
    assert store.cas_update_calls, "cas_update_memory was not called"
    assert store.cas_update_calls[0]["expected_revision"] == phase_a
    assert store.cas_update_calls[0]["expected_revision"] != fresh


@pytest.mark.asyncio
async def test_execute_delete_passes_phase_a_revision_not_fresh_snapshot():
    consumed = _mem(unit_id="obs-del", text="phase-a text")
    later = _mem(unit_id="obs-del", text="fresh under lock")
    phase_a = memory_revision_token(consumed)
    fresh = memory_revision_token(later)
    store = _FakeStore({"obs-del": MemorySnapshot(memory=later, revision=fresh)})
    with (
        patch.object(C, "get_memories", return_value=store),
        patch.object(C, "_delete_observation_history", new=AsyncMock()),
    ):
        await C._execute_delete_action(
            conn=_Conn(),
            bank_id="b",
            observation_id="obs-del",
            expected_revision=phase_a,
        )
    assert store.cas_delete_calls, "cas_delete_memory was not called"
    assert store.cas_delete_calls[0]["expected_revision"] == phase_a
    assert store.cas_delete_calls[0]["expected_revision"] != fresh


@pytest.mark.asyncio
async def test_fold_create_passes_phase_a_target_revision_not_fresh_snapshot():
    twin = _mem(unit_id="twin", text="phase-a twin")
    later = _mem(unit_id="twin", text="fresh twin")
    phase_a = memory_revision_token(twin)
    fresh = memory_revision_token(later)
    store = _FakeStore({"twin": MemorySnapshot(memory=later, revision=fresh)})
    src = uuid.uuid4()
    outcome = C._DedupOutcome(
        best_id="twin",
        merged_text="merged",
        should_merge=True,
        candidate_ids={"twin"},
        candidate_revisions={"twin": phase_a},
    )
    with (
        patch.object(C, "_filter_live_source_memories", new=AsyncMock(return_value=[src])),
    ):
        await C._dedup_fold_create(
            store=store,
            conn=_Conn(),
            memory_engine=object(),
            bank_id="b",
            config=types.SimpleNamespace(),
            outcome=outcome,
            create_source_ids=[src],
            expected_revision=phase_a,
        )
    assert store.cas_fold_calls, "cas_fold_observation was not called"
    assert store.cas_fold_calls[0]["expected_revision"] == phase_a
    assert store.cas_fold_calls[0]["expected_revision"] != fresh


# --------------------------------------------------------------------------- preparation provenance: LLM sees snapshot-backed text/token


@pytest.mark.asyncio
async def test_prepare_stores_token_of_snapshot_shown_to_llm_not_later_row():
    """Phase A must snapshot before the LLM and store THAT token.

    A later store state (post-recall mutation) must not become the prepared token.
    """
    obs_id = str(uuid.uuid4())
    mem_id = str(uuid.uuid4())
    shown = _mem(unit_id=obs_id, text="shown to llm")
    later = _mem(unit_id=obs_id, text="mutated after recall before snapshot")
    shown_rev = memory_revision_token(shown)
    later_rev = memory_revision_token(later)
    assert shown_rev != later_rev

    shown_fact = _obs_fact(obs_id, shown.text)
    fake_recall = types.SimpleNamespace(results=[shown_fact], source_facts={})
    llm_result = C._BatchLLMResult(
        updates=[C._UpdateAction(text="rewritten", observation_id=obs_id, source_fact_ids=[mem_id])]
    )

    # snapshot_memories returns the shown object; a naive post-LLM re-snapshot would
    # be later_rev. The helper under test must store shown_rev.
    store = _FakeStore({obs_id: MemorySnapshot(memory=shown, revision=shown_rev)})

    class _Pool:
        _wraps_backend = True

        @asynccontextmanager
        async def acquire(self):
            yield _Conn()

    mem_engine = types.SimpleNamespace(
        embeddings=object(),
        _consolidation_llm_config=types.SimpleNamespace(with_config=lambda *a, **k: object()),
    )
    captured_union: list = []

    async def _capture_llm(**kwargs):
        captured_union.append(list(kwargs.get("union_observations") or []))
        return llm_result

    with (
        patch.object(C, "_find_related_observations", new=AsyncMock(return_value=fake_recall)),
        patch.object(C, "_consolidate_batch_with_llm", new=_capture_llm),
        patch.object(C, "_effective_scope_limit", return_value=-1),
        patch.object(C, "get_memories", return_value=store),
        patch.object(C, "_dedup_active", return_value=False),
    ):
        prepared = await C._prepare_memory_batch(
            pool=_Pool(),
            memory_engine=mem_engine,
            llm_config=object(),
            bank_id="bank1",
            memories=[{"id": mem_id, "text": "Some memory", "tags": ["t1"]}],
            request_context=object(),
            config=types.SimpleNamespace(consolidation_dedup_threshold=1.0),
        )

    assert prepared.updates, "expected an UPDATE plan"
    assert prepared.updates[0].phase_a_revision == shown_rev
    assert prepared.updates[0].phase_a_revision != later_rev
    assert prepared.observation_revisions[obs_id] == shown_rev
    assert captured_union and captured_union[0][0].text == shown.text
