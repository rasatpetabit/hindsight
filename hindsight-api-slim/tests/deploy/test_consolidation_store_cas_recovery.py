"""Container-native non-SQL witness recovery test (design §6.5 / M5 carry-over).

Drives the REAL ``MaintenanceLoop._run_txn_recovery`` against a durable non-SQL store double
(a SQLite-backed MemoriesExtension that owns its rows outside Postgres — ``writes_memory_rows_in_sql
= False``), proving the recovery invariants the store-CAS redesign inherits:

  - witnessed publishes:  a write-group whose witness row exists is COMMITTED by recovery
  - unwitnessed aborts:   a write-group past grace with no witness is ABORTED
  - idempotent:           repeated recovery leaves observable state unchanged
  - same-fate:            source marks and observation writes share one fate

The store double is a TEST ASSET (mounted from tests/, never shipped); the code under test —
real ``MaintenanceLoop._run_txn_recovery`` and the real base-store txn contract — is always the
image's own. Adapted from the frozen branch's M5 container recovery test.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import tempfile
from pathlib import Path

import pytest

# ── container detection ────────────────────────────────────────────────────────

_APP_ROOT = Path("/app/api/hindsight_api")
_REAL_MAINTENANCE = _APP_ROOT / "engine" / "maintenance.py"
_IN_CONTAINER = _REAL_MAINTENANCE.exists()

if not _IN_CONTAINER:
    pytest.skip(
        "container-gated non-SQL witness recovery test: requires the Hindsight image "
        "(real /app/api/hindsight_api/engine/maintenance.py). Run:\n"
        "  docker run --rm -v \"$PWD/hindsight-api-slim/hindsight_api:/app/api/hindsight_api\" "
        "-v \"$PWD/hindsight-api-slim/tests:/tests:ro\" --entrypoint /app/api/.venv/bin/python "
        "<hindsight-image> -m pytest /tests/deploy/test_consolidation_store_cas_recovery.py -v",
        allow_module_level=True,
    )

from hindsight_api.engine.maintenance import MaintenanceLoop  # noqa: E402


def _assert_real_source() -> None:
    """criterion-1: prove the method under test is the production one."""
    srcfile = inspect.getsourcefile(MaintenanceLoop._run_txn_recovery) or ""
    assert srcfile.startswith("/app/api/"), (
        f"MaintenanceLoop._run_txn_recovery resolves from {srcfile!r}, expected under /app/api/"
    )


# Load the durable SQLite store double from the mounted tests.
_SHM_DURABLE = Path("/tests/deploy/_hindsight_shim/hindsight_api/engine/durable_memory_store.py")
if not _SHM_DURABLE.exists():
    raise RuntimeError("requires /tests/deploy/_hindsight_shim mounted read-only")


def _import_shim():
    import sys as _sys

    name = "hindsight_shim_durable"
    spec = importlib.util.spec_from_file_location(name, _SHM_DURABLE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    _sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        _sys.modules.pop(name, None)
        raise
    return mod


_shim_mod = _import_shim()
DurableMemoryStore = _shim_mod.DurableMemoryStore  # noqa: N816


class DurableStore(DurableMemoryStore):
    """Non-SQL store double satisfying the REAL upstream contract."""

    writes_memory_rows_in_sql = False


# ── fake MemoryEngine backend for MaintenanceLoop ──────────────────────────────


class _BackendConn:
    async def fetch(self, sql):
        return [["coding"]]


class _Backend:
    """Duck-typed engine backend consumed by MaintenanceLoop via acquire_with_retry."""

    async def acquire(self):
        return _BackendConn()

    async def release(self, conn):
        return None

    def get_size(self):
        return 1

    def get_idle_size(self):
        return 0


class _FakeEngineForLoop:
    def __init__(self):
        self._backend = _Backend()


async def _run_real_recovery(store, *, prime_first_seen=None) -> int:
    """Invoke REAL MaintenanceLoop._run_txn_recovery with get_memories() -> our store."""
    import hindsight_api.engine.memories as mem_mod

    engine = _FakeEngineForLoop()
    loop = MaintenanceLoop(engine)
    loop._txn_first_seen.clear()
    if prime_first_seen:
        loop._txn_first_seen.update(prime_first_seen)

    original_get_memories = mem_mod.get_memories
    mem_mod.get_memories = lambda: store
    try:
        await loop._run_txn_recovery()
    finally:
        mem_mod.get_memories = original_get_memories

    return len([s for s in store.pending_txn_states().values() if s in ("committed", "aborted")])


# ── pytest entrypoints ─────────────────────────────────────────────────────────


def test_recovery_real_source_is_production():
    """criterion-1: method under test resolves under /app/api/ (not a shim)."""
    """criterion-1: method under test resolves under /app/api/ (not a shim)."""
    _assert_real_source()


@pytest.mark.asyncio
async def test_non_sql_witness_recovery_publishes():
    """Design §6.5: witnessed write-group → published by REAL recovery; idempotent repeat."""
    from hindsight_api.engine.memories import FactRecord

    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "recovery.db")
        store = DurableStore(db_path)

        # A write-group with a witness (writer committed witness then crashed before decide).
        txn = await store.mint_txn(bank_id="coding", mutating=True)
        await store.upsert_observation(
            bank_id="coding",
            txn=txn,
            record=FactRecord(
                unit_id="u-wit", text="witnessed observation", embedding=None,
                fact_type="observation",
                source_memory_ids=["s-wit"], proof_count=1,
            ),
        )
        await store.mark_consolidated(
            bank_id="coding", unit_ids=["mu-wit"], failed=True, txn=txn,
        )
        await store.write_txn_witness(txn, conn=None, fq_table=None)

        # Pre-recovery: held writes invisible; source not marked success.
        pre_rows = await store.get_memories(bank_id="coding")
        pre_texts = [r.get("text") for r in pre_rows]
        assert "witnessed observation" not in pre_texts, f"held write leaked pre-recovery: {pre_texts}"
        pre_marks = store.marks()
        assert pre_marks.get("mu-wit") is None, f"source marked before recovery: {pre_marks}"
        assert store.witness_count() >= 1

        # REAL recovery decides against witness table.
        await _run_real_recovery(store)

        post_rows = await store.get_memories(bank_id="coding")
        post_texts = [r.get("text") for r in post_rows]
        assert "witnessed observation" in post_texts, f"recovery should publish (got {post_texts})"
        post_marks = store.marks()
        assert post_marks.get("mu-wit") is False, f"failed mark should publish after recovery (got {post_marks})"
        assert store.pending_write_count() == 0, "no pending writes after committed recovery"

        # Idempotent repeat.
        rows_before_repeat = await store.get_memories(bank_id="coding")
        marks_before_repeat = store.marks()
        await _run_real_recovery(store)
        assert rows_before_repeat == await store.get_memories(bank_id="coding")
        assert marks_before_repeat == store.marks()


@pytest.mark.asyncio
async def test_non_sql_witness_recovery_aborts_unwitnessed():
    """Design §6.5: unwitnessed write-group past grace → aborted; idempotent."""
    from hindsight_api.engine.memories import FactRecord

    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "recovery-unwit.db")
        store = DurableStore(db_path)

        txn = await store.mint_txn(bank_id="coding", mutating=True)
        await store.upsert_observation(
            bank_id="coding",
            txn=txn,
            record=FactRecord(
                unit_id="u-unwit", text="never-witnessed", embedding=None,
                fact_type="observation",
                source_memory_ids=["s-unwit"], proof_count=1,
            ),
        )
        await store.mark_consolidated(
            bank_id="coding", unit_ids=["mu-unwit"], failed=True, txn=txn,
        )
        assert store.witness_count() == 0

        pre_rows = await store.get_memories(bank_id="coding")
        assert len(pre_rows) == 0, f"held write leaked pre-recovery: {pre_rows}"

        # Prime first_seen so the txn is past grace on the FIRST sweep.
        first_seen = {txn["id"]: 1000.0}
        await _run_real_recovery(store, prime_first_seen=first_seen)

        post_rows = await store.get_memories(bank_id="coding")
        assert len(post_rows) == 0, f"unwitnessed group should be discarded (got {post_rows})"
        assert store.marks() == {}, f"no mark should publish for aborted group (got {store.marks()})"
        assert store.pending_write_count() == 0, "pending not cleared after abort"

        states_before = store.pending_txn_states()
        await _run_real_recovery(store, prime_first_seen=first_seen)
        assert states_before == store.pending_txn_states(), "repeat changed txn states (non-idempotent)"
