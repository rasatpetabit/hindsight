"""Task 5: Phase A always precomputes embeddings; Phase B with conn never embeds.

When dedup is disabled the embedding currently lives only inside ``if dedup_enabled``,
so Phase B re-embeds under the bank lock. These tests require:

1. ``_prepare_memory_batch`` populates ``embedding_str`` on UPDATE and CREATE even
   when ``_dedup_active`` is False.
2. ``_execute_update_action`` / ``_create_observation_directly`` with a live
   ``conn`` and ``precomputed_embedding=None`` raise rather than calling
   ``generate_embeddings_batch``.
"""

from __future__ import annotations

import types
import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.engine.consolidation import consolidator as C
from hindsight_api.engine.memories.base import MemorySnapshot, StoredMemory, memory_revision_token
from hindsight_api.engine.response_models import MemoryFact


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


class _Pool:
    _wraps_backend = True

    @asynccontextmanager
    async def acquire(self):
        yield _Conn()


class _Store:
    def __init__(self, snapshots):
        self.snapshots = snapshots

    def writes_memory_rows_in_sql_for(self, _bank_id: str) -> bool:
        return True

    async def snapshot_memories(self, *, conn, fq_table, bank_id, unit_ids):
        return [self.snapshots[str(u)] for u in unit_ids if str(u) in self.snapshots]


def _mem(unit_id: str, text: str) -> StoredMemory:
    return StoredMemory(unit_id=unit_id, text=text, fact_type="observation", tags=["t1"])


async def _prepare(*, creates=None, updates=None, memories, union_obs, store):
    llm_result = C._BatchLLMResult(creates=creates or [], updates=updates or [])
    fake_recall = types.SimpleNamespace(results=union_obs, source_facts={})
    embed_calls: list[list[str]] = []

    async def _embed(_backend, texts, **kwargs):
        embed_calls.append(list(texts))
        return ["[0.42]"] * len(texts)

    mem_engine = types.SimpleNamespace(
        embeddings=object(),
        _consolidation_llm_config=types.SimpleNamespace(with_config=lambda *a, **k: object()),
    )
    with (
        patch.object(C, "_find_related_observations", new=AsyncMock(return_value=fake_recall)),
        patch.object(C, "_consolidate_batch_with_llm", new=AsyncMock(return_value=llm_result)),
        patch.object(C, "_effective_scope_limit", return_value=-1),
        patch.object(C, "get_memories", return_value=store),
        patch.object(C, "_dedup_active", return_value=False),
        patch.object(C.embedding_utils, "generate_embeddings_batch", new=_embed),
        patch.object(C, "_dedup_adjudicate", new=AsyncMock(side_effect=AssertionError("dedup must not run"))),
    ):
        prepared = await C._prepare_memory_batch(
            pool=_Pool(),
            memory_engine=mem_engine,
            llm_config=object(),
            bank_id="bank1",
            memories=memories,
            request_context=object(),
            config=types.SimpleNamespace(consolidation_dedup_threshold=1.0),
        )
    return prepared, embed_calls


@pytest.mark.asyncio
async def test_prepare_update_embeds_when_dedup_disabled():
    obs_id = str(uuid.uuid4())
    mem_id = str(uuid.uuid4())
    shown = _mem(obs_id, "existing observation")
    store = _Store({obs_id: MemorySnapshot(memory=shown, revision=memory_revision_token(shown))})
    prepared, embed_calls = await _prepare(
        updates=[C._UpdateAction(text="rewritten observation", observation_id=obs_id, source_fact_ids=[mem_id])],
        memories=[{"id": mem_id, "text": "source fact", "tags": ["t1"]}],
        union_obs=[MemoryFact(id=obs_id, text=shown.text, fact_type="observation", tags=["t1"])],
        store=store,
    )
    assert prepared.updates, "expected an UPDATE plan"
    assert prepared.updates[0].embedding_str == "[0.42]"
    assert prepared.updates[0].dedup_outcome is None
    assert embed_calls == [["rewritten observation"]]


@pytest.mark.asyncio
async def test_prepare_create_embeds_when_dedup_disabled():
    mem_id = str(uuid.uuid4())
    store = _Store({})
    prepared, embed_calls = await _prepare(
        creates=[C._CreateAction(text="brand new observation", source_fact_ids=[mem_id])],
        memories=[{"id": mem_id, "text": "source fact", "tags": ["t1"]}],
        union_obs=[],
        store=store,
    )
    assert prepared.creates, "expected a CREATE plan"
    assert prepared.creates[0].embedding_str == "[0.42]"
    assert prepared.creates[0].dedup_outcome is None
    assert embed_calls == [["brand new observation"]]


@pytest.mark.asyncio
async def test_execute_update_with_conn_raises_instead_of_embedding():
    """Phase B (conn provided) must not call the embedder when precomputed_embedding is missing."""
    embed = AsyncMock(side_effect=AssertionError("embedder must not run under the bank lock"))
    src = uuid.uuid4()
    fact = MemoryFact(id="obs-upd", text="old", fact_type="observation", tags=["t1"])
    with (
        patch.object(C.embedding_utils, "generate_embeddings_batch", new=embed),
        patch.object(C, "get_memories", return_value=_Store({})),
        patch.object(C, "_filter_live_source_memories", new=AsyncMock(return_value=[src])),
    ):
        with pytest.raises(ValueError, match="precomputed_embedding"):
            await C._execute_update_action(
                pool=object(),
                memory_engine=types.SimpleNamespace(
                    embeddings=object(),
                    _backend=types.SimpleNamespace(ops=types.SimpleNamespace(uses_observation_sources_table=False)),
                ),
                bank_id="b",
                source_memory_ids=[src],
                observation_id="obs-upd",
                new_text="rewritten",
                observations=[fact],
                conn=_Conn(),
                precomputed_embedding=None,
                expected_revision="phase-a",
            )
    embed.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_observation_with_conn_raises_instead_of_embedding():
    """Phase B CREATE must not call the embedder when precomputed_embedding is missing."""
    embed = AsyncMock(side_effect=AssertionError("embedder must not run under the bank lock"))
    src = uuid.uuid4()
    with (
        patch.object(C.embedding_utils, "generate_embeddings_batch", new=embed),
        patch.object(C, "get_memories", return_value=_Store({})),
        patch.object(C, "_filter_live_source_memories", new=AsyncMock(return_value=[src])),
        patch.object(C, "_any_live_source_memory", new=AsyncMock(return_value=True)),
    ):
        with pytest.raises(ValueError, match="precomputed_embedding"):
            await C._create_observation_directly(
                pool=object(),
                memory_engine=types.SimpleNamespace(
                    embeddings=object(),
                    _backend=types.SimpleNamespace(ops=types.SimpleNamespace(uses_observation_sources_table=False)),
                ),
                bank_id="b",
                source_memory_ids=[src],
                observation_text="new observation",
                conn=_Conn(),
                precomputed_embedding=None,
            )
    embed.assert_not_awaited()
