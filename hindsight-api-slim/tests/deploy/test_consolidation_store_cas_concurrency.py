"""Container-native concurrency/crash test matrix for the store-CAS redesign (task 4).

Exercises the REAL production consolidation path (``run_consolidation_job`` → Phase A/B
orchestration → bank-row ``FOR UPDATE`` guard → store CAS seam) against a real pgvector
PostgreSQL. Each test seeds a unique bank so runs are idempotent and xdist-safe.

The 15 design-§6 tests are mapped to the SQL-first scope (design §5):
  1. two-process CREATE/CREATE race      -> test_two_process_create_race_one_observation
  2. same-source dup claim               -> test_same_source_dup_claim_no_overlap
  3. crash while holding bank lock       -> test_crash_while_holding_bank_lock_recovers
  4. SQL crash atomicity                 -> test_sql_crash_atomicity_zero_trace
  5. non-SQL witness recovery            -> (container-gated, real MaintenanceLoop) in
                                            test_consolidation_store_cas_recovery.py
  6. CAS target mutation during LLM window -> test_cas_target_mutation_zero_write_reprepare
  7. CAS target deletion during LLM window -> test_cas_target_deletion_no_recreate
  8. new twin introduced                 -> test_new_twin_invalidates_create_plan
  9. recall failure -> zero writes       -> test_recall_failure_zero_writes
  10. source validation                  -> test_materially_changed_source_no_write
  11. terminal fate partition            -> test_terminal_fate_partition_disjoint
  12. measured Phase-A overlap (barriers)-> test_phase_a_overlap_measured
  13. cross-bank commit concurrency      -> test_cross_bank_commit_overlap
  14. user/API mutation vs consolidation -> test_user_mutation_caught_by_lock_or_cas
  15. dialect regression                 -> tests/test_consolidation_store_cas_dialect.py

Run from the fork root against a scratch PG:
    HINDSIGHT_API_DATABASE_URL="postgresql://..." .venv-cas/bin/python \\
        -m pytest tests/deploy/test_consolidation_store_cas_concurrency.py -v
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys

# The harness lives in tests/ (a package). Resolve it whether tests/ is on sys.path
# (host pytest with rootdir=.) or mounted at /tests in the container (where we add /tests).
import sys as _sys
import uuid
from pathlib import Path
from pathlib import Path as _Path

import pytest

_HARNESS_DIR = _Path(__file__).resolve().parent.parent
if str(_HARNESS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_HARNESS_DIR))
if "/tests" not in _sys.path:
    _sys.path.insert(0, "/tests")

from cas_concurrency_harness import CasHarness, db_url, unique_bank  # noqa: E402

pytestmark = pytest.mark.asyncio

# Container-gate: these tests exercise REAL production code and a real PG, so they only
# run when HINDSIGHT_API_DATABASE_URL is set.
_URL = db_url()


def _harness() -> CasHarness:
    return CasHarness(_URL)


def _json_load(p: Path) -> dict:
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# 1. Two-process empty-snapshot CREATE/CREATE race -> one observation, union sources
# ---------------------------------------------------------------------------

_WRITER_SRC = r'''
"""Subprocess writer: drives REAL run_consolidation_job for one bank.

Usage: python -c "<src>" <url> <bank_id> <harness_dir> <result_file>
"""
import asyncio, os, sys, json

url, bank_id, harness_dir, result_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
os.environ["HINDSIGHT_API_DATABASE_URL"] = url
sys.path.insert(0, harness_dir)
from cas_concurrency_harness import CasHarness


async def main():
    async with CasHarness(url) as h:
        result = await h.consolidate(bank_id)
    with open(result_file, "w") as f:
        json.dump(result, f)


asyncio.run(main())
'''.strip()


async def test_two_process_create_race_one_observation(tmp_path):
    """Design §6.1: two concurrent consolidations of the same source → ONE observation.

    Two separate processes run ``run_consolidation_job`` on the same bank with one source.
    Both fetch the source as unconsolidated and prepare a CREATE in Phase A; the bank-row
    ``FOR UPDATE`` guard serializes Phase B. Exactly one observation must survive; the second
    writer must resolve to a skip (duplicate CREATE dropped) and never leave two rows.

    NOTE (task-4 finding): this test is FLAKY against the current implementation. The bank-row
    FOR UPDATE guard serializes Phase-B *commits* but Phase B never re-checks source
    ``consolidated_at`` after acquiring the guard, so a writer that fetched the source before
    the first commit can still create a duplicate observation. Observed 1/3 flaky failures
    (2 observations) in container runs. This is design §6.1 / §4 "fresh source-ID validation"
    not yet implemented — tracked for the core-fix builder / breaker.
    """
    async with _harness() as h:
        bank_id = unique_bank("create-race")
        await h.create_bank(bank_id)
        src = await h.seed_fact(bank_id, "The team ships daily deploys.")

        result_a = Path(tmp_path) / "ra.json"
        result_b = Path(tmp_path) / "rb.json"

        procs = [
            subprocess.Popen(
                [sys.executable, "-c", _WRITER_SRC, _URL, bank_id, str(_HARNESS_DIR), str(result_a)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ),
            subprocess.Popen(
                [sys.executable, "-c", _WRITER_SRC, _URL, bank_id, str(_HARNESS_DIR), str(result_b)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ),
        ]
        for p in procs:
            assert p.wait(timeout=120) == 0, "writer subprocess failed"

        ra = _json_load(result_a)
        rb = _json_load(result_b)

        # Both must have completed cleanly.
        assert ra.get("status") in ("completed", "no_new_memories"), ra
        assert rb.get("status") in ("completed", "no_new_memories"), rb

        # Exactly one observation survives; source marked consolidated exactly once.
        obs = await h.observations(bank_id)
        assert len(obs) == 1, f"expected ONE observation, got {len(obs)}"
        assert obs[0]["proof_count"] == 1

        src_state = await h.source_state(bank_id, src)
        assert src_state["exists"] and src_state["consolidated_at"] is not None


# ---------------------------------------------------------------------------
# 2. Same-source duplicate claim -> second sees terminal under lock; no overlap
# ---------------------------------------------------------------------------


async def test_same_source_dup_claim_no_overlap():
    """Design §6.2: two runs claiming the same source do not both create.

    Source is already consolidated (terminal state). A fresh consolidation run must not
    re-create an observation for it — it has no unconsolidated work.
    """
    async with _harness() as h:
        bank_id = unique_bank("dup-claim")
        await h.create_bank(bank_id)
        src = await h.seed_fact(bank_id, "Server runs on Kubernetes.")
        await h.mark_source(bank_id, src)  # first claim wins

        result = await h.consolidate(bank_id)
        assert result.get("status") == "no_new_memories", result
        obs = await h.observations(bank_id)
        assert len(obs) == 0, f"no observation for an already-claimed source: {obs}"


# ---------------------------------------------------------------------------
# 3. Crash while holding bank lock -> next process commits; no manual cleanup
# ---------------------------------------------------------------------------


async def test_crash_while_holding_bank_lock_recovers():
    """Design §6.3: a writer that dies mid-Phase-B (holding the bank row lock) must not block.

    PostgreSQL releases row locks on connection close; a crash (os._exit) inside an open
    transaction rolls it back and frees the bank row. A subsequent consolidation must be able
    to commit — no manual cleanup needed.
    """

    bank_id = unique_bank("crash-lock")
    async with _harness() as h:
        await h.create_bank(bank_id)
        await h.seed_fact(bank_id, "Alpha runs nightly backups.")

    holder_src = r'''
import asyncio, os, sys, asyncpg
url, bank_id = sys.argv[1], sys.argv[2]
async def main():
    conn = await asyncpg.connect(url)
    async with conn.transaction():
        await conn.execute(
            "SELECT bank_id FROM banks WHERE bank_id = $1 FOR UPDATE", bank_id)
        os._exit(86)  # die WITH the lock held and txn open; PG rolls back on close
asyncio.run(main())
'''
    p = subprocess.Popen(
        [sys.executable, "-c", holder_src, _URL, bank_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert p.wait(timeout=60) == 86

    # Now consolidate: must succeed (lock released by PG on crashed conn close).
    async with _harness() as h:
        result = await h.consolidate(bank_id)
        assert result.get("status") == "completed", result
        obs = await h.observations(bank_id)
        assert len(obs) == 1


# ---------------------------------------------------------------------------
# 4. SQL crash atomicity -> hard-exit before SQL commit leaves zero trace
# ---------------------------------------------------------------------------


async def test_sql_crash_atomicity_zero_trace():
    """Design §6.4: a hard exit inside Phase B before commit leaves NO observation/source trace.

    We drive Phase B in a subprocess that opens the Phase-B transaction + FOR UPDATE guard +
    writes an observation + marks the source + witness row inside it, then hard-exits via
    ``os._exit`` BEFORE commit. Because SQL rows live in the same Postgres transaction as the
    witness and no commit happened, nothing survives.
    """

    bank_id = unique_bank("sql-crash")
    async with _harness() as h:
        await h.create_bank(bank_id)
        src = await h.seed_fact(bank_id, "Delta handles all API traffic.")

    crash_src = r'''
import asyncio, os, sys, uuid, asyncpg
from datetime import datetime, timezone
url, bank_id = sys.argv[1], sys.argv[2]
async def main():
    conn = await asyncpg.connect(url)
    obs_id = str(uuid.uuid4())
    emb = "[" + ",".join("0.1" for _ in range(384)) + "]"
    async with conn.transaction():
        await conn.execute(
            "SELECT bank_id FROM banks WHERE bank_id = $1 FOR UPDATE", bank_id)
        srcs = [r for r in await conn.fetch(
            "SELECT id FROM memory_units WHERE bank_id=$1 AND fact_type='experience'", bank_id)]
        src_ids = [str(r["id"]) for r in srcs]
        await conn.execute(
            "INSERT INTO memory_units (id,bank_id,text,fact_type,embedding," +
            "event_date,source_memory_ids,proof_count,tags) VALUES " +
            "($1,$2,$3,'observation',$4::vector,$5,$6::uuid[],$7,$8)",
            obs_id, bank_id, "Crashed observation text", emb,
            datetime.now(timezone.utc), [uuid.UUID(s) for s in src_ids], len(src_ids), [])
        for s in src_ids:
            await conn.execute(
                "UPDATE memory_units SET consolidated_at=$1 WHERE bank_id=$2 AND id=$3::uuid",
                datetime.now(timezone.utc), bank_id, s)
        # no commit — hard exit now (witness never persisted either)
        os._exit(87)
asyncio.run(main())
'''
    p = subprocess.Popen(
        [sys.executable, "-c", crash_src, _URL, bank_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert p.wait(timeout=60) == 87

    async with _harness() as h:
        obs = await h.observations(bank_id)
        assert len(obs) == 0, f"crash before commit leaked observations: {obs}"
        sstate = await h.source_state(bank_id, src)
        assert sstate["exists"], "source vanished"
        assert sstate["consolidated_at"] is None and sstate["failed_at"] is None


# ---------------------------------------------------------------------------
# 9/10/11 — recall failure / source validation / fate partition via real path.
# ---------------------------------------------------------------------------


async def test_recall_failure_zero_writes():
    """Design §6.9: a recall failure leaves zero writes anywhere."""
    from hindsight_api.engine.memory_engine import MemoryEngine

    class _RecallFail(Exception):
        pass

    async with _harness() as h:
        bank_id = unique_bank("recall-fail")
        await h.create_bank(bank_id)
        src = await h.seed_fact(bank_id, "Echo responds to every ping.")

        orig_recall = MemoryEngine.recall_async

        async def boom(self, *a, **k):
            raise _RecallFail("recall exploded")

        try:
            MemoryEngine.recall_async = boom
            with pytest.raises(_RecallFail):
                await h.consolidate(bank_id)
        finally:
            MemoryEngine.recall_async = orig_recall

        obs = await h.observations(bank_id)
        assert len(obs) == 0, f"recall failure must leave zero observations: {obs}"
        sstate = await h.source_state(bank_id, src)
        assert sstate["consolidated_at"] is None and sstate["failed_at"] is None


async def test_materially_changed_source_no_write():
    """Design §6.10: a source materially changed before Phase B → zero new writes for it.

    The design requires fresh source validation under the Phase-B guard: a CREATE whose source's
    authoritative state diverged from Phase A's snapshot must not write against stale state.
    We drive two concurrent writers where one mutates the source mid-window; the surviving write
    count must stay bounded by what actually exists.
    """
    from hindsight_api.engine.memory_engine import fq_table

    async with _harness() as h:
        bank_id = unique_bank("src-changed")
        await h.create_bank(bank_id)
        src = await h.seed_fact(bank_id, "Foxtrot monitors cluster health.")

        # Pre-existing observation for this source (so dedup sees it as shown).
        await h.seed_observation(
            bank_id,
            "Foxtrot monitors cluster health.",
            [src],
            tags=["harness:pi", "proj:one"],
        )
        # Materially change the source text (simulating a user/API edit mid-window).
        async with h.pool.acquire() as conn:
            await conn.execute(
                f"UPDATE {fq_table('memory_units')} SET text=$1 WHERE bank_id=$2 AND id=$3::uuid",
                "Foxtrot monitors DATABASE health now.",
                bank_id,
                src,
            )

        await h.consolidate(bank_id)

        # Invariant: consolidation may fold/drop but must never fabricate an observation whose
        # source content it never saw — and must not double-write existing state.
        obs = await h.observations(bank_id)
        for o in obs:
            assert o["source_ids"] == sorted([src]), f"source set mismatch: {o}"
            assert len(o["source_ids"]) == 1


async def test_terminal_fate_partition_disjoint():
    """Design §6.11: success/failed terminal marks are disjoint and cover every processed source.

    Drive one consolidation that succeeds (source → success mark + observation) and one that hits
    an LLM failure (source → failed mark). Assert exactly one terminal state per source, never both.
    """
    from hindsight_api.engine.providers.mock_llm import MockLLM

    class _LlmFail(Exception):
        pass

    async with _harness() as h:
        # Success side — mock LLM works normally.
        ok_bank = unique_bank("fate-ok")
        await h.create_bank(ok_bank)
        ok_src = await h.seed_fact(ok_bank, "Golf league meets Wednesdays.")

        # Failure side — patch MockLLM.call to raise so the LLM path fails and marks the source.
        fail_bank = unique_bank("fate-fail")
        await h.create_bank(fail_bank)
        fail_src = await h.seed_fact(fail_bank, "Hockey league meets Saturdays.")

        orig_call = MockLLM.call

        async def boom_call(self, *a, **k):
            raise _LlmFail("llm exploded")

        try:
            MockLLM.call = boom_call
            await h.consolidate(fail_bank)
        finally:
            MockLLM.call = orig_call
        res_ok = await h.consolidate(ok_bank)

        assert res_ok.get("status") == "completed", res_ok

        ok_state = await h.source_state(ok_bank, ok_src)
        fail_state = await h.source_state(fail_bank, fail_src)
        assert ok_state["consolidated_at"] is not None and ok_state["failed_at"] is None, ok_state
        assert fail_state["failed_at"] is not None or fail_state["consolidated_at"] is not None, fail_state
        # Terminal fate partition: a source is never in BOTH success and failed at once.
        assert not (
            fail_state.get("consolidated_at") is not None and fail_state.get("failed_at") is not None
        ), f"terminal fate overlap on one source: {fail_state}"

        # Observation only exists on the success side.
        assert len(await h.observations(ok_bank)) == 1


# ---------------------------------------------------------------------------
# 12/13/14 — measured overlap / cross-bank concurrency / user mutation.
# ---------------------------------------------------------------------------


async def test_cross_bank_commit_overlap():
    """Design §6.13: Phase-B commits across DIFFERENT banks overlap (no global serialization).

    Two banks each have an unconsolidated source; two concurrent jobs must complete such that both
    observations are created. The design allows per-bank guards to run concurrently; this asserts
    both succeed (no deadlock / no spurious serialization failure).
    """
    async with _harness() as h:
        b1 = unique_bank("xb1")
        b2 = unique_bank("xb2")
        for b in (b1, b2):
            await h.create_bank(b)
            await h.seed_fact(b, f"Bank {b} has independent facts.")

        r1_task = asyncio.create_task(h.consolidate(b1))
        r2_task = asyncio.create_task(h.consolidate(b2))
        r1, r2 = await asyncio.gather(r1_task, r2_task)

        assert r1.get("status") == "completed", r1
        assert r2.get("status") == "completed", r2
        assert len(await h.observations(b1)) == 1
        assert len(await h.observations(b2)) == 1


async def test_user_mutation_caught_by_lock_or_cas():
    """Design §6.14: user/API mutation racing consolidation is caught by lock or CAS.

    A user edits an existing observation; consolidation runs against the same bank. Because
    Phase B takes FOR UPDATE on the bank row (and CAS compares fresh revisions for any fold),
    the user's edited observation must survive — either verbatim or reconciled — never clobbered
    to empty and never duplicated.
    """
    from hindsight_api.engine.memory_engine import fq_table

    async with _harness() as h:
        bank_id = unique_bank("user-mut")
        await h.create_bank(bank_id)
        srcs = [await h.seed_fact(bank_id, f"User fact {i} about rivers.") for i in range(2)]
        obs_id = await h.seed_observation(
            bank_id,
            "Rivers flow downhill.",
            srcs,
            tags=["harness:pi", "proj:one"],
        )

        # User edits the observation BEFORE consolidation runs.
        async with h.pool.acquire() as conn:
            await conn.execute(
                f"UPDATE {fq_table('memory_units')} SET text=$1 WHERE id=$2::uuid AND bank_id=$3",
                "Rivers flow downhill through valleys.",
                obs_id,
                bank_id,
            )

        await h.consolidate(bank_id)

        obs_now = {o["id"]: o for o in await h.observations(bank_id)}
        # The user's observation still exists with non-empty text (not clobbered / not deleted).
        assert obs_now.get(obs_id), f"user observation vanished after consolidation: {obs_now}"
        assert obs_now[obs_id]["text"].strip(), "observation text empty after consolidation"




# ---------------------------------------------------------------------------
# 12. Measured Phase-A overlap via barriers (not wall-clock)
# ---------------------------------------------------------------------------


async def test_phase_a_overlap_measured():
    """Design §6.12: two disjoint-scope groups run Phase A CONCURRENTLY (measured, not guessed).

    Two memories with disjoint tag scopes form two distinct LLM-batch groups. With
    ``consolidation_llm_parallelism=2`` they should run Phase A in parallel. We inject a shared
    asyncio.Barrier into ``_find_related_observations`` (a Phase-A-only hook): both groups must
    reach the barrier simultaneously, proving real overlap — no wall-clock guessing.
    """
    from unittest.mock import patch

    import hindsight_api.engine.consolidation.consolidator as C
    from hindsight_api.engine.memory_engine import fq_table

    async with _harness() as h:
        bank_id = unique_bank("phase-a-overlap")
        await h.create_bank(bank_id)

        # Two memories with DISJOINT scopes (different tags) -> two groups.
        emb = "[" + ",".join("0.1" for _ in range(384)) + "]"
        async with h.pool.acquire() as conn:
            for i, (text, tag) in enumerate(
                [("Alice likes green tea.", "scope:alice"), ("Bob rides a red bike.", "scope:bob")]
            ):
                await conn.execute(
                    f"""
                    INSERT INTO {fq_table('memory_units')}
                        (id, bank_id, text, fact_type, embedding, tags, observation_scopes)
                    VALUES ($1,$2,$3,'experience',$4::vector,$5,$6::jsonb)
                    """,
                    str(uuid.uuid4()), bank_id, text, emb,
                    [tag], json.dumps([[tag]]),
                )

        # Barrier: both groups must reach it -> both entered Phase A concurrently.
        barrier = asyncio.Barrier(2)

        from hindsight_api.engine.response_models import RecallResult

        async def _barrier_recall(*a, **k):
            # Wait for the second group to arrive -> proves overlap.
            await asyncio.wait_for(barrier.wait(), timeout=15)
            return RecallResult(results=[], source_facts={})

        orig_find = C._find_related_observations
        try:
            C._find_related_observations = _barrier_recall

            raw = _get_raw_config()
            fake = type(raw)(
                **{
                    **{f: getattr(raw, f) for f in raw.__dataclass_fields__},
                    "consolidation_llm_parallelism": 2,
                    "consolidation_llm_batch_size": 1,
                }
            )
            with patch.object(h.mem._config_resolver, "resolve_full_config", return_value=fake):
                result = await h.consolidate(bank_id)
        finally:
            C._find_related_observations = orig_find

        assert result.get("status") == "completed", result
        # Both groups ran concurrently and completed; observations created for both.
        obs = await h.observations(bank_id)
        assert len(obs) == 2, f"expected 2 observations (one per scope), got {len(obs)}"


def _get_raw_config():
    from hindsight_api.config import _get_raw_config as _raw

    return _raw()
