"""Shared harness for the store-CAS concurrency/crash test matrix (task 4).

Builds a lightweight real ``MemoryEngine`` (mock LLM provider, stubbed embeddings, no
ML model loading) against a pgvector PostgreSQL reachable via ``HINDSIGHT_API_DATABASE_URL``,
and drives the REAL production consolidation path (``run_consolidation_job``) so the tests
exercise the exact Phase A/B orchestration, bank-row ``FOR UPDATE`` guard, and store CAS seam
that the design mandates — never a shim.

Usage (container-native, run inside the Hindsight image or with source overlaid):

    docker run --rm \\
        -v "$PWD/hindsight-api-slim/hindsight_api:/app/api/hindsight_api" \\
        -v "$PWD/hindsight-api-slim/tests:/tests:ro" \\
        -e HINDSIGHT_API_DATABASE_URL="postgresql://..." \\
        --entrypoint /app/api/.venv/bin/python \\
        <hindsight-image> -m pytest /tests/deploy/test_consolidation_store_cas_concurrency.py -v

or via the repo-local venv (no container):

    HINDSIGHT_API_DATABASE_URL="postgresql://..." .venv-cas/bin/python \\
        -m pytest tests/deploy/test_consolidation_store_cas_concurrency.py -v

Every function is idempotent over a unique ``bank_id`` so tests can run under xdist or
repeatedly against the same scratch DB without cross-test contamination.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from hindsight_api import MemoryEngine, RequestContext
from hindsight_api.engine.task_backend import SyncTaskBackend

# pgvector dimension fixed by the migrated schema (384d model). All seeded embeddings use it.
EMB_DIM = 384


class FakeEmbeddings:
    """Duck-typed embeddings stub: no ML model, deterministic vectors."""

    provider_name = "mock"
    dimension = EMB_DIM

    async def initialize(self) -> None:
        pass


class FakeCrossEncoder:
    """Duck-typed cross-encoder stub: no ML model."""

    blocking_init = False

    async def initialize(self) -> None:
        pass


async def _fake_embed(embeddings_obj, texts, **kwargs):
    """Deterministic 384-dim vector strings; accepts any caller kwargs (input_type etc.)."""
    vec = "[" + ",".join("0.1" for _ in range(EMB_DIM)) + "]"
    return [vec] * len(texts)


def install_fake_embeddings() -> None:
    """Patch the module-level embedding call sites used by consolidation + recall.

    The consolidator imports ``embedding_utils`` at module scope; recall reaches the same
    function through ``embedding_processing``/``embedding_utils``. Patch both so no ML
    model is ever loaded.
    """
    from hindsight_api.engine.retain import embedding_utils as retain_utils

    retain_utils.generate_embeddings_batch = _fake_embed

    import hindsight_api.engine.consolidation.consolidator as C

    C.embedding_utils.generate_embeddings_batch = _fake_embed

    try:
        from hindsight_api.engine import embedding_processing

        embedding_processing.generate_embeddings_batch = _fake_embed
    except Exception:
        pass


def db_url() -> str:
    url = os.getenv("HINDSIGHT_API_DATABASE_URL")
    if not url:
        raise RuntimeError(
            "HINDSIGHT_API_DATABASE_URL not set — point at a pgvector PostgreSQL "
            "(e.g. postgresql://postgres:postgres@<host>:5432/hindsight)"
        )
    return url


class CasHarness:
    """One lightweight real MemoryEngine over the scratch PG + helpers to seed state.

    Each instance owns a fresh backend pool and exposes store/CAS helpers so tests can
    seed banks/memories/observations and read back authoritative state directly.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self.mem: MemoryEngine | None = None

    async def __aenter__(self) -> "CasHarness":
        install_fake_embeddings()
        self.mem = MemoryEngine(
            db_url=self.url,
            memory_llm_provider="mock",
            memory_llm_api_key="",
            memory_llm_model="mock",
            embeddings=FakeEmbeddings(),
            cross_encoder=FakeCrossEncoder(),
            pool_min_size=1,
            pool_max_size=10,
            run_migrations=False,
            task_backend=SyncTaskBackend(),
            skip_llm_verification=True,
        )
        await self.mem.initialize()
        return self

    async def __aexit__(self, *exc) -> None:
        if self.mem is not None:
            try:
                await self.mem.close()
            except Exception:
                pass
            self.mem = None

    @property
    def pool(self):
        return self.mem._backend

    def request_context(self) -> RequestContext:
        return RequestContext(internal=True)

    # ------------------------------------------------------------------ seeding

    async def create_bank(self, bank_id: str) -> None:
        from hindsight_api.engine.memory_engine import fq_table

        async with self.pool.acquire() as conn:
            await conn.execute(
                f"INSERT INTO {fq_table('banks')} (bank_id, name) VALUES ($1, $2)",
                bank_id,
                "cas-concurrency-test",
            )

    async def seed_fact(
        self,
        bank_id: str,
        text: str,
        *,
        tags: list[str] | None = None,
        fact_type: str = "experience",
    ) -> str:
        """Insert an unconsolidated source fact; returns its unit id."""
        from hindsight_api.engine.memory_engine import fq_table

        unit_id = str(uuid.uuid4())
        emb = "[" + ",".join("0.1" for _ in range(EMB_DIM)) + "]"
        async with self.pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {fq_table('memory_units')}
                    (id, bank_id, text, fact_type, embedding, tags)
                VALUES ($1,$2,$3,$4,$5::vector,$6)
                """,
                unit_id,
                bank_id,
                text,
                fact_type,
                emb,
                tags or ["harness:pi", "proj:one"],
            )
        return unit_id

    async def seed_observation(
        self,
        bank_id: str,
        text: str,
        source_ids: list[str],
        *,
        tags: list[str] | None = None,
    ) -> str:
        """Insert an observation row directly (as if created by a prior consolidation)."""
        from datetime import datetime, timezone

        from hindsight_api.engine.memory_engine import fq_table

        obs_id = str(uuid.uuid4())
        emb = "[" + ",".join("0.2" for _ in range(EMB_DIM)) + "]"
        async with self.pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {fq_table('memory_units')}
                    (id, bank_id, text, fact_type, embedding, event_date,
                     source_memory_ids, proof_count, tags)
                VALUES ($1,$2,$3,'observation',$4::vector,$5,$6::uuid[],$7,$8)
                """,
                obs_id,
                bank_id,
                text,
                emb,
                datetime.now(timezone.utc),
                [uuid.UUID(s) for s in source_ids],
                len(source_ids),
                tags or [],
            )
        return obs_id

    async def mark_source(self, bank_id: str, unit_id: str, *, failed: bool = False) -> None:
        """Mark a source consolidated (or failed), simulating a completed consolidation."""
        from datetime import datetime, timezone

        from hindsight_api.engine.memory_engine import fq_table

        col = "consolidation_failed_at" if failed else "consolidated_at"
        async with self.pool.acquire() as conn:
            await conn.execute(
                f"UPDATE {fq_table('memory_units')} SET {col} = $1 WHERE bank_id=$2 AND id=$3::uuid",
                datetime.now(timezone.utc),
                bank_id,
                unit_id,
            )

    # ------------------------------------------------------------------ reads

    async def observations(self, bank_id: str) -> list[dict[str, Any]]:
        from hindsight_api.engine.memory_engine import fq_table

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, text, proof_count, source_memory_ids
                FROM {fq_table('memory_units')}
                WHERE bank_id=$1 AND fact_type='observation'
                """,
                bank_id,
            )
            return [
                {
                    "id": str(r["id"]),
                    "text": r["text"],
                    "proof_count": r["proof_count"],
                    "source_ids": sorted(str(s) for s in (r["source_memory_ids"] or [])),
                }
                for r in rows
            ]

    async def source_state(self, bank_id: str, unit_id: str) -> dict[str, Any]:
        from hindsight_api.engine.memory_engine import fq_table

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT id, text, consolidated_at, consolidation_failed_at
                FROM {fq_table('memory_units')}
                WHERE bank_id=$1 AND id=$2::uuid
                """,
                bank_id,
                unit_id,
            )
            if row is None:
                return {"exists": False}
            return {
                "exists": True,
                "id": str(row["id"]),
                "text": row["text"],
                "consolidated_at": row["consolidated_at"],
                "failed_at": row["consolidation_failed_at"],
            }

    # ------------------------------------------------------------------ drive

    async def consolidate(
        self,
        bank_id: str,
        *,
        observation_scopes=None,
        pending_refresh_tags=None,
        operation_id=None,
    ) -> dict[str, Any]:
        """Run the REAL consolidation job and return its result dict."""
        from hindsight_api.engine.consolidation.consolidator import run_consolidation_job

        return await run_consolidation_job(
            self.mem,
            bank_id,
            self.request_context(),
            operation_id=operation_id,
            observation_scopes=observation_scopes,
            pending_refresh_tags=pending_refresh_tags,
        )


def unique_bank(prefix: str = "cas") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"
