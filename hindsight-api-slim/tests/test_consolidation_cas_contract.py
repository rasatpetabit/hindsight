"""Pure tests for the consolidation store-native snapshot + CAS contract (design §4.3).

These tests exercise the seam's *contract* without a live database:

1. **Revision token determinism + mutation sensitivity** — ``memory_revision_token``
   must be stable for an unchanged authoritative state and change when any
   consolidation-relevant field changes (text, source lineage, tags, temporal,
   fact type, context, metadata, proof count, consolidated markers). It must NOT
   change on ``updated_at``/``created_at`` bookkeeping writes (the design's stated
   rationale for excluding them).
2. **CAS outcome semantics** — the ``CASOutcome`` enum and the base method
   signatures/docs define applied/stale/missing distinctly; the pure surface is
   verified here.
3. **Fail-closed base defaults** — a store that does not implement the seam (no
   genuine optimistic concurrency) must raise ``CASNotSupportedError`` rather than
   silently pretend a blind write succeeded.

Postgres integration (snapshot/update/delete/fold applied-vs-stale against a real
database) is covered separately in task 1b. This file deliberately touches no DB.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from hindsight_api.engine.memories.base import (
    CASNotSupportedError,
    CASOutcome,
    MemoriesExtension,
    MemorySnapshot,
    StoredMemory,
    memory_revision_token,
)


def _mem(**overrides) -> StoredMemory:
    """A canonical observation with stable defaults; callers override authoritative fields."""
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


# --------------------------------------------------------------------------- token


def test_revision_token_deterministic_for_equal_state():
    assert memory_revision_token(_mem()) == memory_revision_token(_mem())


def test_revision_token_stable_under_unrelated_field_changes():
    """Bookkeeping fields that consolidation does not act on must NOT change the token."""
    base = memory_revision_token(_mem())
    # updated_at is deliberately excluded (pg/reads.mark_consolidated leaves it alone).
    assert memory_revision_token(_mem(created_at=datetime(2024, 2, 1, tzinfo=timezone.utc))) == base


@pytest.mark.parametrize(
    "mutator",
    [
        lambda: dict(text="A completely different observation text."),
        lambda: dict(fact_type="memory"),
        lambda: dict(context=None),
        lambda: dict(tags=["only"]),
        lambda: dict(tags=["programming"]),  # order-insensitive set membership: one dropped tag
        lambda: dict(source_memory_ids=["s1"]),
        lambda: dict(source_memory_ids=["s2", "s1"]),  # order-independent -> same as [s1,s2]
        lambda: dict(metadata={"other": "x"}),
        lambda: dict(proof_count=3),
        lambda: dict(event_date=None),
        lambda: dict(occurred_start=datetime(2023, 12, 31, tzinfo=timezone.utc)),
        lambda: dict(occurred_end=None),
        lambda: dict(mentioned_at=None),
        lambda: dict(observation_scopes=["harness:cli"]),
        lambda: dict(consolidated_at=datetime(2024, 2, 2, tzinfo=timezone.utc)),
    ],
)
def test_revision_token_sensitive_to_each_authoritative_field(mutator):
    base = memory_revision_token(_mem())
    mutated = _mem(**mutator())
    # Sanity: the mutator actually changed something material.
    if "tags" in mutator() and sorted(mutator()["tags"]) == sorted(["programming", "history"]):
        pytest.skip("order-only permutation is intentionally stable")
    if "source_memory_ids" in mutator() and sorted(mutator()["source_memory_ids"]) == ["s1", "s2"]:
        pytest.skip("order-only permutation is intentionally stable")
    assert memory_revision_token(mutated) != base


def test_revision_token_source_order_insensitive():
    """Source lineage is a set; permuting order must not change the token."""
    a = _mem(source_memory_ids=["s1", "s2"])
    b = _mem(source_memory_ids=["s2", "s1"])
    assert memory_revision_token(a) == memory_revision_token(b)


def test_revision_token_tag_order_insensitive():
    a = _mem(tags=["programming", "history"])
    b = _mem(tags=["history", "programming"])
    assert memory_revision_token(a) == memory_revision_token(b)


def test_snapshot_carries_memory_and_revision():
    snap = MemorySnapshot(memory=_mem(), revision=memory_revision_token(_mem()))
    assert snap.memory.text == "Ada designed the first algorithm."
    assert snap.revision == memory_revision_token(snap.memory)


# --------------------------------------------------------------------------- outcomes


def test_cas_outcome_semantics():
    """APPLIED / STALE / MISSING are distinct; nothing is written for STALE or MISSING."""
    assert CASOutcome.APPLIED.value == "applied"
    assert CASOutcome.STALE.value == "stale"
    assert CASOutcome.MISSING.value == "missing"
    assert len({CASOutcome.APPLIED, CASOutcome.STALE, CASOutcome.MISSING}) == 3


# --------------------------------------------------------------------------- fail-closed base


def _make_no_cas_store():
    """A concrete store that inherits the base (non-CAS) seam defaults.

    The base methods raise :class:`CASNotSupportedError` by design; this store only
    exists so we can instantiate the abstract :class:`MemoriesExtension`.
    """

    def _noop(*_a, **_k):
        raise NotImplementedError("abstract-method stub; never called in these tests")

    abstracts = sorted(MemoriesExtension.__abstractmethods__)
    return type("NoCASStore", (MemoriesExtension,), {n: _noop for n in abstracts})(config={})


@pytest.mark.asyncio
async def test_base_snapshot_fails_closed():
    store = _make_no_cas_store()
    with pytest.raises(CASNotSupportedError):
        await store.snapshot_memories(
            conn=None, fq_table=lambda n: n, bank_id="b", unit_ids=["u"]
        )


@pytest.mark.asyncio
async def test_base_cas_update_fails_closed():
    store = _make_no_cas_store()
    with pytest.raises(CASNotSupportedError):
        await store.cas_update_memory(
            conn=None,
            fq_table=lambda n: n,
            bank_id="b",
            unit_id="u",
            expected_revision="r",
            patch=None,
        )


@pytest.mark.asyncio
async def test_base_cas_delete_fails_closed():
    store = _make_no_cas_store()
    with pytest.raises(CASNotSupportedError):
        await store.cas_delete_memory(
            conn=None,
            fq_table=lambda n: n,
            bank_id="b",
            unit_id="u",
            expected_revision="r",
        )


@pytest.mark.asyncio
async def test_base_cas_fold_fails_closed():
    store = _make_no_cas_store()
    with pytest.raises(CASNotSupportedError):
        await store.cas_fold_observation(
            conn=None,
            fq_table=lambda n: n,
            bank_id="b",
            observation_id="u",
            expected_revision="r",
            merged_text="t",
        )
