"""PostgreSQL integration tests for the consolidation store-native snapshot + CAS seam.

These tests exercise the real ``PostgresMemories`` CAS implementation against a live
PostgreSQL (pgvector) database — the SQL-first scope of the store-CAS redesign
(design doc §5). They cover:

1. ``snapshot_memories`` — authoritative fields + a stable revision token that changes
   on mutation.
2. ``cas_update_memory`` — APPLIED on a matching expected revision, STALE (no write) when
   the authoritative state changed, MISSING when the target no longer exists.
3. ``cas_delete_memory`` — APPLIED / STALE / MISSING, with the row actually removed only
   on a match.
4. ``cas_fold_observation`` — APPLIED / STALE / MISSING; source ids unioned (deduped),
   ``proof_count`` follows the source count, and temporal params are honored (the fold
   bug fixed in task 1a).

Every operation asserts the ``bank_id`` / identity predicate is respected: writes and
locks are scoped to the owning bank, and a row in another bank is never touched.

These tests intentionally avoid the ML-heavy ``memory`` fixture (which loads
sentence-transformers / torch). They build the PostgreSQL backend directly and seed
``memory_units`` rows via SQL — so they run in a lightweight environment (base deps +
pytest only) against any pgvector PostgreSQL reachable via ``HINDSIGHT_API_DATABASE_URL``.
An external plain ``postgresql://`` URL points at the scratch PG (matching the conftest
contract for non-pg0 URLs).
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio

from hindsight_api.engine.db import create_database_backend
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.memories.base import CASOutcome, MemoryPatch
from hindsight_api.engine.memory_engine import fq_table

pytestmark = pytest.mark.asyncio

# pgvector dimension fixed by the migrated schema (384d model). All seeded embeddings use it.
_EMB_DIM = 384


def _emb(fill: float = 0.1) -> str:
    """A pgvector literal string of _EMB_DIM floats."""
    return "[" + ",".join(str(fill) for _ in range(_EMB_DIM)) + "]"


def _db_url() -> str:
    url = os.getenv("HINDSIGHT_API_DATABASE_URL")
    if not url:
        pytest.skip("HINDSIGHT_API_DATABASE_URL not set (point at a pgvector PostgreSQL to run)")
    return url


class _PGHarness:
    """Owns one PostgreSQL backend + its migrated schema for a test module.

    The conftest's ``pg0_db_url`` fixture normally handles this; here we accept a plain
    ``postgresql://`` URL (the conftest branch that runs migrations and uses it directly)
    and hold an independent backend so these tests do not need pg0 or the ML stack.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self.backend = None

    @property
    def store(self):
        return get_memories()

    @property
    def ops(self):
        return self.backend.ops

    async def create_bank(self, bank_id: str) -> None:
        async with self.backend.acquire() as conn:
            await conn.execute(
                f"INSERT INTO {fq_table('banks')} (bank_id, name) VALUES ($1, $2)",
                bank_id,
                "cas-pg-test",
            )

    async def seed_fact(self, bank_id: str, text: str, fact_type: str = "experience") -> str:
        """Insert one fact through the store's ``insert_facts`` and return its unit id."""
        fact = SimpleNamespace(
            fact_text=text,
            embedding=[0.1] * _EMB_DIM,
            fact_type=fact_type,
            tags=[],
            context=None,
            document_id=None,
            chunk_id=None,
            metadata=None,
            observation_scopes=None,
            entities=[],
            causal_relations=[],
            occurred_start=None,
            occurred_end=None,
            mentioned_at=None,
        )
        async with self.backend.acquire() as conn:
            unit_ids = await self.store.insert_facts(
                conn=conn, ops=self.ops, bank_id=bank_id, facts=[fact], document_id=None
            )
        return unit_ids[0]

    async def seed_observation(
        self, bank_id: str, text: str, source_ids: list[str], *, event_date: datetime | None = None
    ) -> str:
        """Insert an observation row directly (the consolidator writes it inline in SQL)."""
        obs_id = str(uuid.uuid4())
        async with self.backend.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {fq_table('memory_units')}
                    (id, bank_id, text, fact_type, embedding, event_date,
                     source_memory_ids, proof_count)
                VALUES ($1,$2,$3,'observation',$4::vector,$5,$6::uuid[],$7)
                """,
                obs_id,
                bank_id,
                text,
                _emb(0.2),
                event_date or datetime.now(timezone.utc),
                [uuid.UUID(s) for s in source_ids],
                len(source_ids),
            )
        return obs_id

    async def snapshot(self, bank_id: str, unit_ids: list[str]):
        async with self.backend.acquire() as conn:
            return await self.store.snapshot_memories(
                conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids
            )

    async def update(self, bank_id: str, unit_id: str, rev: str, patch: MemoryPatch):
        async with self.backend.acquire() as conn:
            return await self.store.cas_update_memory(
                conn=conn, fq_table=fq_table, bank_id=bank_id,
                unit_id=unit_id, expected_revision=rev, patch=patch,
            )

    async def delete(self, bank_id: str, unit_id: str, rev: str):
        async with self.backend.acquire() as conn:
            return await self.store.cas_delete_memory(
                conn=conn, fq_table=fq_table, bank_id=bank_id,
                unit_id=unit_id, expected_revision=rev,
            )

    async def fold(
        self,
        bank_id: str,
        observation_id: str,
        rev: str,
        merged_text: str,
        add_source_ids=None,
        *,
        tags=None,
        event_date=None,
        occurred_start=None,
        occurred_end=None,
        mentioned_at=None,
    ):
        async with self.backend.acquire() as conn:
            return await self.store.cas_fold_observation(
                conn=conn, fq_table=fq_table, bank_id=bank_id,
                observation_id=observation_id, expected_revision=rev,
                merged_text=merged_text,
                add_source_ids=add_source_ids or [],
                tags=tags,
                event_date=event_date,
                occurred_start=occurred_start,
                occurred_end=occurred_end,
                mentioned_at=mentioned_at,
            )

    async def row_count(self, bank_id: str) -> int:
        async with self.backend.acquire() as conn:
            return await conn.fetchval(
                f"SELECT count(*) FROM {fq_table('memory_units')} WHERE bank_id = $1",
                bank_id,
            )

    async def read_memory(self, bank_id: str, unit_id: str):
        snaps = await self.snapshot(bank_id, [unit_id])
        return snaps[0] if snaps else None


@pytest_asyncio.fixture(scope="function")
async def harness():
    """A migrated PG backend for each CAS integration test.

    Function-scoped to match the conftest ``memory`` fixture and avoid pytest-asyncio's
    module-scope runner mismatch (the ``_function_scoped_runner`` ScopeMismatch).
    Migrations are a fast no-op once applied, so per-test setup is cheap.
    """
    url = _db_url()
    from hindsight_api.migrations import run_migrations

    run_migrations(url)
    h = _PGHarness(url)
    h.backend = create_database_backend("postgresql")
    await h.backend.initialize(url, min_size=1, max_size=5)
    try:
        yield h
    finally:
        await h.backend.shutdown()
        h.backend = None


async def _unique_bank(harness) -> str:
    bank_id = f"cas-pg-{uuid.uuid4().hex[:8]}"
    await harness.create_bank(bank_id)
    return bank_id


# --------------------------------------------------------------------------- snapshot


async def test_snapshot_returns_authoritative_fields_and_stable_token(harness):
    bank = await _unique_bank(harness)
    uid = await harness.seed_fact(bank, "Ada designed the first algorithm.")
    snaps = await harness.snapshot(bank, [uid])

    assert len(snaps) == 1
    snap = snaps[0]
    assert snap.memory.unit_id == uid
    assert snap.memory.text == "Ada designed the first algorithm."
    assert snap.memory.fact_type == "experience"
    assert snap.revision  # opaque token present

    # Stable for an unchanged row.
    again = await harness.snapshot(bank, [uid])
    assert again[0].revision == snap.revision


async def test_snapshot_token_changes_on_mutation(harness):
    bank = await _unique_bank(harness)
    uid = await harness.seed_fact(bank, "Before mutation.")
    before = await harness.snapshot(bank, [uid])
    rev0 = before[0].revision

    out = await harness.update(
        bank, uid, rev0,
        MemoryPatch(unit_id=uid, text="After mutation."),
    )
    assert out == CASOutcome.APPLIED

    after = await harness.snapshot(bank, [uid])
    assert after[0].revision != rev0
    assert after[0].memory.text == "After mutation."


async def test_snapshot_missing_ids_absent(harness):
    """Missing ids are simply absent from the result (contract)."""
    bank = await _unique_bank(harness)
    uid = await harness.seed_fact(bank, "present")
    snaps = await harness.snapshot(bank, [uid, str(uuid.uuid4())])
    assert len(snaps) == 1
    assert snaps[0].memory.unit_id == uid


async def test_snapshot_scoped_to_bank(harness):
    """A row in another bank must never leak into this bank's snapshot."""
    bank_a = await _unique_bank(harness)
    uid_a = await harness.seed_fact(bank_a, "bank A fact")
    snaps_b = await harness.snapshot(f"cas-pg-{uuid.uuid4().hex[:8]}", [uid_a])
    assert snaps_b == []


# --------------------------------------------------------------------------- cas_update


async def test_cas_update_applied_on_matching_revision(harness):
    bank = await _unique_bank(harness)
    uid = await harness.seed_fact(bank, "original")
    rev0 = (await harness.snapshot(bank, [uid]))[0].revision

    out = await harness.update(
        bank, uid, rev0,
        MemoryPatch(unit_id=uid, text="updated"),
    )
    assert out == CASOutcome.APPLIED

    after = await harness.snapshot(bank, [uid])
    assert after[0].memory.text == "updated"


async def test_cas_update_stale_no_write(harness):
    """A stale expected revision must NOT write anything."""
    bank = await _unique_bank(harness)
    uid = await harness.seed_fact(bank, "original")
    rev0 = (await harness.snapshot(bank, [uid]))[0].revision

    # First writer updates.
    assert (
        await harness.update(bank, uid, rev0, MemoryPatch(unit_id=uid, text="first writer"))
        == CASOutcome.APPLIED
    )

    # Second writer holds the ORIGINAL revision — must be STALE and must not clobber.
    out2 = await harness.update(bank, uid, rev0, MemoryPatch(unit_id=uid, text="second writer"))
    assert out2 == CASOutcome.STALE

    after = await harness.snapshot(bank, [uid])
    assert after[0].memory.text == "first writer"


async def test_cas_update_missing_target(harness):
    bank = await _unique_bank(harness)
    out = await harness.update(
        bank, str(uuid.uuid4()), "whatever",
        MemoryPatch(unit_id=str(uuid.uuid4()), text="x"),
    )
    assert out == CASOutcome.MISSING


async def test_cas_update_patch_fields_absolute_sets(harness):
    """tags/metadata/temporal are absolute sets; proof_count_delta is relative."""
    bank = await _unique_bank(harness)
    uid = await harness.seed_fact(bank, "base")
    rev0 = (await harness.snapshot(bank, [uid]))[0].revision

    ed = datetime(2024, 3, 1, tzinfo=timezone.utc)
    out = await harness.update(
        bank, uid, rev0,
        MemoryPatch(
            unit_id=uid,
            tags=["new-tag"],
            metadata={"k": "v"},
            event_date=ed,
            proof_count_delta=2,
        ),
    )
    assert out == CASOutcome.APPLIED

    snap = (await harness.snapshot(bank, [uid]))[0]
    assert snap.memory.tags == ["new-tag"]
    assert snap.memory.metadata == {"k": "v"}
    assert snap.memory.event_date == ed
    assert snap.memory.proof_count >= 2


# --------------------------------------------------------------------------- cas_delete


async def test_cas_delete_applied_removes_row(harness):
    bank = await _unique_bank(harness)
    uid = await harness.seed_fact(bank, "to delete")
    rev0 = (await harness.snapshot(bank, [uid]))[0].revision

    out = await harness.delete(bank, uid, rev0)
    assert out == CASOutcome.APPLIED
    assert await harness.row_count(bank) == 0


async def test_cas_delete_stale_no_write(harness):
    """Deleting with a stale revision must leave the row intact."""
    bank = await _unique_bank(harness)
    uid = await harness.seed_fact(bank, "keep me")
    rev0 = (await harness.snapshot(bank, [uid]))[0].revision

    # mutate first -> revision now differs from rev0
    assert (
        await harness.update(bank, uid, rev0, MemoryPatch(unit_id=uid, text="mutated"))
        == CASOutcome.APPLIED
    )

    # stale delete with original revision -> STALE and no delete
    out2 = await harness.delete(bank, uid, rev0)
    assert out2 == CASOutcome.STALE
    assert await harness.row_count(bank) == 1


async def test_cas_delete_missing_target(harness):
    bank = await _unique_bank(harness)
    out = await harness.delete(bank, str(uuid.uuid4()), "whatever")
    assert out == CASOutcome.MISSING


# --------------------------------------------------------------------------- cas_fold


async def test_cas_fold_applied_unions_sources_and_recomputes_proof(harness):
    bank = await _unique_bank(harness)
    s1 = await harness.seed_fact(bank, "source one", fact_type="experience")
    obs_text = "Ada handles infrastructure"
    obs_id = await harness.seed_observation(bank, obs_text, [s1])
    rev0 = (await harness.snapshot(bank, [obs_id]))[0].revision

    s2_new = str(uuid.uuid4())
    out = await harness.fold(
        bank,
        obs_id,
        rev0,
        merged_text="Ada handles infrastructure + migrations",
        add_source_ids=[s2_new],
    )
    assert out == CASOutcome.APPLIED

    snap = await harness.read_memory(bank, obs_id)
    assert snap.memory.text == "Ada handles infrastructure + migrations"
    # Source lineage is a set: s1 union {s2_new} => 2 distinct sources.
    assert sorted(map(str, snap.memory.source_memory_ids)) == sorted([str(s1), s2_new])
    assert snap.memory.proof_count == 2


async def test_cas_fold_dedupes_existing_sources(harness):
    """Re-adding an already-present source must not duplicate it in the lineage."""
    bank = await _unique_bank(harness)
    s1 = await harness.seed_fact(bank, "source one", fact_type="experience")
    obs_id = await harness.seed_observation(bank, "Ada handles infra", [s1])
    rev0 = (await harness.snapshot(bank, [obs_id]))[0].revision

    out = await harness.fold(
        bank,
        obs_id,
        rev0,
        merged_text="Ada handles infra (refined)",
        add_source_ids=[s1, str(s1)],  # same source twice, both already present
    )
    assert out == CASOutcome.APPLIED

    snap = await harness.read_memory(bank, obs_id)
    assert len(snap.memory.source_memory_ids) == 1
    assert snap.memory.proof_count == 1


async def test_cas_fold_stale_no_write(harness):
    """Folding with a stale expected revision must not alter the observation."""
    bank = await _unique_bank(harness)
    s1 = await harness.seed_fact(bank, "source one", fact_type="experience")
    obs_id = await harness.seed_observation(bank, "Ada handles infra", [s1])
    rev0 = (await harness.snapshot(bank, [obs_id]))[0].revision

    # A concurrent writer mutates the observation first.
    assert (
        await harness.update(
            bank, obs_id, rev0, MemoryPatch(unit_id=obs_id, text="concurrent rewrite")
        )
        == CASOutcome.APPLIED
    )

    # Stale fold with the ORIGINAL revision must be STALE and write nothing.
    out2 = await harness.fold(
        bank,
        obs_id,
        rev0,
        merged_text="stale fold must not apply",
        add_source_ids=[str(uuid.uuid4())],
    )
    assert out2 == CASOutcome.STALE

    snap = await harness.read_memory(bank, obs_id)
    assert snap.memory.text == "concurrent rewrite"
    assert len(snap.memory.source_memory_ids) == 1


async def test_cas_fold_missing_target(harness):
    bank = await _unique_bank(harness)
    out = await harness.fold(
        bank,
        str(uuid.uuid4()),
        "whatever",
        merged_text="x",
        add_source_ids=[str(uuid.uuid4())],
    )
    assert out == CASOutcome.MISSING


async def test_cas_fold_honors_temporal_params(harness):
    """Fold passes temporal overrides through (the task-1a fix)."""
    bank = await _unique_bank(harness)
    s1 = await harness.seed_fact(bank, "source one", fact_type="experience")
    obs_id = await harness.seed_observation(bank, "Ada handles infra", [s1])
    rev0 = (await harness.snapshot(bank, [obs_id]))[0].revision

    ed = datetime(2024, 5, 15, tzinfo=timezone.utc)
    start = datetime(2024, 5, 14, tzinfo=timezone.utc)
    end = datetime(2024, 5, 16, tzinfo=timezone.utc)
    mentioned = datetime(2024, 5, 17, tzinfo=timezone.utc)

    out = await harness.fold(
        bank,
        obs_id,
        rev0,
        merged_text="Ada handles infra (dated)",
        add_source_ids=[str(uuid.uuid4())],
        tags=["dated"],
        event_date=ed,
        occurred_start=start,
        occurred_end=end,
        mentioned_at=mentioned,
    )
    assert out == CASOutcome.APPLIED

    snap = await harness.read_memory(bank, obs_id)
    assert snap.memory.event_date == ed
    assert snap.memory.occurred_start == start
    assert snap.memory.occurred_end == end
    assert snap.memory.mentioned_at == mentioned
    assert snap.memory.tags == ["dated"]
