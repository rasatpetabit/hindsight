"""Deterministic unit tests for the consolidation duplicate-create guard.

These exercise the dedup decision directly (no LLM, no DB), so they reliably
guard the fix in CI — unlike the real-LLM integration test, which only triggers
the path stochastically.
"""

import logging
import types
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from unittest.mock import DEFAULT, AsyncMock, patch

import pytest

from hindsight_api.engine.consolidation.consolidator import (
    _DEDUP_PROMPT,
    _dedup_active,
    _dedup_decision_from_response,
    _dedup_reconcile_create,
    _dedup_reconcile_update,
    _DedupDecision,
    _duplicate_create_target,
    _norm_obs_text,
)
from hindsight_api.engine.memories import CASOutcome, RecallArms
from hindsight_api.engine.search.types import RetrievalResult


@dataclass
class _FakeObs:
    id: str
    text: str


def _shown(*observations: _FakeObs) -> dict[str, _FakeObs]:
    return {_norm_obs_text(o.text): o for o in observations}


def test_norm_obs_text_collapses_whitespace_preserves_case() -> None:
    # Whitespace (incl. newlines) collapses; case is preserved.
    assert _norm_obs_text("  The  User  likes BASIL.\n") == "The User likes BASIL."
    assert _norm_obs_text(None) == ""


def test_create_matching_shown_observation_is_duplicate() -> None:
    shown = _shown(_FakeObs(id="11111111-aaaa", text="User waters the herbs early in the morning."))
    # Same text with only-whitespace differences still matches.
    target = _duplicate_create_target("User waters the   herbs early in the morning.", shown, set())
    assert target is not None
    assert target.startswith("shown observation 11111111")


def test_create_differing_only_in_case_is_not_duplicate() -> None:
    # Case-folding would lose information (e.g. acronyms), so a case-only difference
    # is treated as novel rather than silently dropped.
    shown = _shown(_FakeObs(id="22222222-bbbb", text="The user prefers TLS."))
    assert _duplicate_create_target("The user prefers tls.", shown, set()) is None


def test_create_matching_inresponse_update_is_duplicate() -> None:
    update_texts = {_norm_obs_text("Mint is kept in its own separate bed.")}
    target = _duplicate_create_target("Mint is kept in its own separate bed.", {}, update_texts)
    assert target == "an UPDATE in this response"


def test_novel_create_is_not_duplicate() -> None:
    shown = _shown(_FakeObs(id="22222222-bbbb", text="User waters the herbs early in the morning."))
    assert _duplicate_create_target("Rosemary is drought-tolerant.", shown, set()) is None
    assert _duplicate_create_target("", {}, set()) is None


# ── semantic dedup (_dedup_reconcile_create) ──────────────────────────────────
#
# Mocks the embedder, the obs-anchored ANN probe, and the LLM so the decision logic is
# tested without a DB or a real model.

_TWIN_ID = "33333333-3333-4333-8333-333333333333"


def _obs(text: str, sim: float, oid: str = _TWIN_ID) -> RetrievalResult:
    return RetrievalResult(id=oid, text=text, fact_type="observation", similarity=sim)


def _snap(unit_id, sources=(), revision="rev-1"):
    """A MemorySnapshot-shaped fake: only the fields consolidation reads (revision,
    memory.source_memory_ids) are modeled."""
    return types.SimpleNamespace(
        memory=types.SimpleNamespace(source_memory_ids=list(sources)),
        revision=revision,
    )


class _CasConn:
    """Backend-shaped conn for dedup-fold tests in the CAS-seam world.

    The live-source filter still runs as a FOR SHARE SELECT on the connection inside the
    write-group transaction (SQL store path); snapshot + fold + delete now go through the
    store's CAS seam (mock store), not raw SQL on this conn.
    """

    def __init__(self):
        self.active = 0  # >0 while a connection is acquired (set by _CasBackend.acquire)
        self._in_txn = False
        self.live_rows = None  # override liveness rows; None -> echo all source ids as live
        self.fetch = AsyncMock(side_effect=self._fetch)
        self.fetchval = AsyncMock()
        self.fetchrow = AsyncMock()
        self.execute = AsyncMock()

    @asynccontextmanager
    async def transaction(self):
        assert self.active > 0, "fold transaction opened without an acquired connection"
        self._in_txn = True
        try:
            yield
        finally:
            self._in_txn = False

    async def _fetch(self, query, source_ids, bank_id):
        assert self._in_txn, "live-source filter must run inside the write-group transaction"
        assert "FOR SHARE" in query, "live-source filter must hold FOR SHARE on the source rows"
        if self.live_rows is not None:
            return self.live_rows
        return [{"id": s} for s in source_ids]


class _CasBackend:
    """Backend-shaped stand-in matching acquire_with_retry's ``_wraps_backend`` path."""

    _wraps_backend = True

    def __init__(self, conn):
        self._conn = conn

    @asynccontextmanager
    async def acquire(self):
        self._conn.active += 1
        try:
            yield self._conn
        finally:
            self._conn.active -= 1


def _make_dedup_llm(conn):
    """An LLM stub that asserts no pooled connection is held when it is called."""
    llm = types.SimpleNamespace(call=AsyncMock())

    def _assert_released(*a, **k):
        assert conn.active == 0, "no pooled connection may be held during the dedup LLM call"
        return DEFAULT  # fall through to the llm.call.return_value the test sets

    llm.call.side_effect = _assert_released
    return llm


def _ctx(threshold: float = 0.97):
    """Return (kwargs, store_mock) for a _dedup_reconcile_create call.

    The store is patched at ``get_memories`` and carries the task-1 CAS seam
    (snapshot_memories / cas_fold_observation / cas_delete_memory) plus recall_unified and the
    SQL-path flag. ``conn`` rides on the backend used by ``pool`` for the live-source filter.
    """
    conn = _CasConn()
    store = _make_cas_store()
    llm = _make_dedup_llm(conn)
    kwargs = dict(
        pool=_CasBackend(conn),
        memory_engine=types.SimpleNamespace(embeddings=object()),
        bank_id="bank1",
        # The merge path builds a search_vector UPDATE clause from the text-search
        # config, so these must be present (production defaults: native/english).
        config=types.SimpleNamespace(
            consolidation_dedup_threshold=threshold,
            text_search_extension="native",
            text_search_extension_native_language="english",
        ),
        dedup_llm_config=llm,
        create_text="YouTube content in Uzbek is very rich.",
        create_source_ids=[uuid.uuid4()],
        tags=["t1"],
    )
    return kwargs, store, conn


def _make_cas_store():
    """A fake memories store exposing the task-1 CAS seam for dedup-fold tests.

    SQL-path by default (``writes_memory_rows_in_sql_for`` True). Default fold/snapshot/delete
    outcomes are APPLIED / present / APPLIED; individual tests override them.
    """
    store = types.SimpleNamespace(
        recall_unified=AsyncMock(),
        snapshot_memories=AsyncMock(return_value=[_snap(_TWIN_ID)]),
        cas_fold_observation=AsyncMock(return_value=CASOutcome.APPLIED),
        cas_delete_memory=AsyncMock(return_value=CASOutcome.APPLIED),
        writes_sql=True,
    )

    def _writes_sql(bank_id):
        return store.writes_sql

    store.writes_memory_rows_in_sql_for = _writes_sql
    return store


def _patch_probe(results, store):
    """Wire ``results`` into ``store.recall_unified`` (the obs-anchored ANN probe)."""
    store.recall_unified.return_value = {"observation": RecallArms(semantic=results)}
    return _patch_store(store)


def _patch_store(store):
    """Patch get_memories to return a specific CAS-seam store fake.

    Patching is applied at BOTH namespaces because the consolidator binds the module-level
    ``from ..memories import get_memories`` reference (fold paths) while ``_dedup_adjudicate``
    re-imports it locally from the package (recall probe). One namespace alone would leak the
    real store into the other path.
    """
    from contextlib import ExitStack
    from unittest.mock import patch as _patch

    _stack = ExitStack()
    _stack.enter_context(_patch("hindsight_api.engine.memories.get_memories", lambda: store))
    _stack.enter_context(
        _patch("hindsight_api.engine.consolidation.consolidator.get_memories", lambda: store)
    )
    return _stack


def _patch_embed():
    return patch(
        "hindsight_api.engine.retain.embedding_utils.generate_embeddings_batch",
        AsyncMock(return_value=[[0.1, 0.2, 0.3]]),
    )


async def test_dedup_no_twin_above_threshold_returns_none() -> None:
    kwargs, store, conn = _ctx(threshold=0.97)
    llm = kwargs["dedup_llm_config"]
    with _patch_embed(), _patch_probe([_obs("something loosely related", 0.81)], store):
        result = await _dedup_reconcile_create(**kwargs)
    assert result is None
    llm.call.assert_not_called()  # below threshold → no LLM call
    store.cas_fold_observation.assert_not_called()  # no merge


async def test_dedup_llm_keep_does_not_merge() -> None:
    kwargs, store, conn = _ctx()
    llm = kwargs["dedup_llm_config"]
    llm.call.return_value = '{"action": "keep", "text": "", "reason": "different language"}'
    with _patch_embed(), _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.98)], store):
        result = await _dedup_reconcile_create(**kwargs)
    assert result is None
    llm.call.assert_awaited_once()
    store.cas_fold_observation.assert_not_called()  # kept distinct → no merge


async def test_dedup_llm_missing_action_defaults_to_keep() -> None:
    kwargs, store, conn = _ctx()
    llm = kwargs["dedup_llm_config"]
    llm.call.return_value = _DedupDecision(reason="underfilled structured response")
    with _patch_embed(), _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.98)], store):
        result = await _dedup_reconcile_create(**kwargs)
    assert result is None
    llm.call.assert_awaited_once()
    store.cas_fold_observation.assert_not_called()  # missing action is a conservative no-merge


def test_dedup_decision_accepts_exact_valid_actions() -> None:
    assert _DedupDecision(action="merge").action == "merge"
    assert _DedupDecision(action="keep").action == "keep"


def test_dedup_decision_invalid_action_defaults_to_keep(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        decision = _DedupDecision(action="need_input", reason="model asked for more context")

    assert decision.action == "keep"
    assert "need_input" in caplog.text
    assert "defaulting to keep" in caplog.text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Case / whitespace variants of the CORRECT verdict are recovered via
        # normalize, not discarded — a genuine merge must not become a missed merge.
        ("Merge", "merge"),
        (" MERGE ", "merge"),
        ("keep\n", "keep"),
        ("KEEP", "keep"),
        # Unrecognized / non-str values still degrade to keep (unchanged fail-safe;
        # the warning path is covered by the dedicated tests below).
        ("await", "keep"),
        ("unknown", "keep"),
        (None, "keep"),
        (123, "keep"),
    ],
)
def test_dedup_decision_normalizes_action_case_and_whitespace(raw: object, expected: str) -> None:
    assert _DedupDecision(action=raw).action == expected


def test_dedup_decision_non_scalar_action_defaults_to_keep(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        list_decision = _DedupDecision(action=[])
        dict_decision = _DedupDecision(action={"value": "merge"})

    assert list_decision.action == "keep"
    assert dict_decision.action == "keep"
    assert "defaulting to keep" in caplog.text


def test_dedup_decision_accepts_raw_json_and_dict_responses() -> None:
    raw_merge = '{"action": "merge", "text": "Merged observation.", "reason": "same fact"}'
    raw_keep = {"action": "keep", "text": "", "reason": "different fact"}

    merge_decision = _dedup_decision_from_response(raw_merge)
    keep_decision = _dedup_decision_from_response(raw_keep)

    assert merge_decision.action == "merge"
    assert merge_decision.text == "Merged observation."
    assert keep_decision.action == "keep"
    assert keep_decision.text == ""


def test_dedup_decision_legacy_raw_text_defaults_to_keep(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        decision = _dedup_decision_from_response('action="merge" text="Merged observation."')

    assert decision.action == "keep"
    assert decision.reason == "invalid structured response"
    assert "Invalid consolidation dedup response" in caplog.text


def test_dedup_prompt_contract_requests_json_not_key_value() -> None:
    prompt = _DEDUP_PROMPT.format(new="The agent checked health at 14:07.", existing="Health was checked.")

    assert '{"action": "merge", "text": "...", "reason": "..."}' in prompt
    assert '{"action": "keep", "text": "", "reason": "..."}' in prompt
    assert '"text" to an empty string' in prompt
    assert "Do NOT use key=value" in prompt
    assert 'respond action="merge"' not in prompt
    assert "{new}" not in prompt
    assert "{existing}" not in prompt


async def test_dedup_llm_merge_folds_into_twin() -> None:
    kwargs, store, conn = _ctx()
    kwargs["create_source_ids"] = [uuid.uuid4(), uuid.uuid4()]
    llm = kwargs["dedup_llm_config"]
    llm.call.return_value = (
        '{"action": "merge", "text": "Uzbek content on YouTube is very rich.", "reason": "same fact"}'
    )
    with _patch_embed(), _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.99)], store):
        result = await _dedup_reconcile_create(**kwargs)
    assert result == _TWIN_ID  # merged into the twin; caller skips the CREATE
    # Fresh snapshot of the twin precedes the CAS fold (revision-token gate).
    store.snapshot_memories.assert_awaited_once()
    store.cas_fold_observation.assert_awaited_once()
    fold_kwargs = store.cas_fold_observation.await_args.kwargs
    assert fold_kwargs["merged_text"] == "Uzbek content on YouTube is very rich."  # merged text persisted
    assert fold_kwargs["add_source_ids"] == [str(s) for s in kwargs["create_source_ids"]]  # new (live) sources folded in
    assert fold_kwargs["observation_id"] == _TWIN_ID  # onto the twin row
    assert store.cas_fold_observation.return_value == CASOutcome.APPLIED


async def test_dedup_llm_merge_sanitizes_text_before_write() -> None:
    # The merge path writes the LLM's synthesized text straight to the CAS fold, so it needs
    # the same character-safety scrub _CreateAction/_UpdateAction already apply via field_validator.
    # A raw NUL reaching the driver breaks the Postgres UTF-8 encode.
    kwargs, store, conn = _ctx()
    kwargs["create_source_ids"] = [uuid.uuid4(), uuid.uuid4()]
    llm = kwargs["dedup_llm_config"]
    llm.call.return_value = _DedupDecision(
        action="merge", text="Uzbek content\x00 on YouTube is very rich.", reason="same fact"
    )
    with _patch_embed(), _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.99)], store):
        result = await _dedup_reconcile_create(**kwargs)
    assert result == _TWIN_ID  # still folds into the twin
    store.cas_fold_observation.assert_awaited_once()
    folded_text = store.cas_fold_observation.await_args.kwargs["merged_text"]
    assert "\x00" not in folded_text  # the control character never reaches SQL
    assert folded_text == "Uzbek content on YouTube is very rich."  # scrubbed, not mangled


async def test_dedup_picks_highest_above_threshold_skips_below() -> None:
    # Only the >=threshold candidate is considered; a 0.95 result is ignored at threshold 0.97.
    kwargs, store, conn = _ctx(threshold=0.97)
    llm = kwargs["dedup_llm_config"]
    llm.call.return_value = _DedupDecision(action="keep")
    with _patch_embed(), _patch_probe([_obs("near but distinct", 0.95), _obs("the real twin", 0.98)], store):
        await _dedup_reconcile_create(**kwargs)
    # the twin passed to the LLM is the >=0.97 one, not the 0.95
    sent = llm.call.await_args.kwargs["messages"][0]["content"]
    assert "the real twin" in sent
    assert "near but distinct" not in sent


# ── UPDATE-path dedup (_dedup_reconcile_update) ───────────────────────────────
#
# An UPDATE rewrites+re-embeds an observation, which can drift it into a near-twin of a
# DIFFERENT existing observation. These cover the fold-and-delete reconciliation (unlike
# CREATE, both rows already exist), the self-exclusion, and the keep/no-twin no-ops.

_UPDATED_ID = "44444444-4444-4444-8444-444444444444"


def _update_ctx(threshold: float = 0.97, updated_sources=None):
    """Return (kwargs, store, conn) for a _dedup_reconcile_update call."""
    conn = _CasConn()
    store = _make_cas_store()
    # snapshot_memories returns per-id snapshots; side_effect maps [twin] / [updated] lookups.
    store.snapshot_memories.side_effect = None
    store.snapshot_memories.return_value = None  # placeholder; overridden per call in the fold
    if updated_sources is None:
        updated_sources = [uuid.uuid4(), uuid.uuid4()]
    store._updated_sources = updated_sources

    def _snapshots(conn=None, fq_table=None, bank_id=None, unit_ids=None):
        # unit_ids is a list with one id: return the twin or the updated row's snapshot.
        if unit_ids == [_TWIN_ID]:
            return [_snap(_TWIN_ID)]
        return [_snap(unit_ids[0], sources=store._updated_sources)]

    store.snapshot_memories.side_effect = _snapshots
    llm = _make_dedup_llm(conn)
    kwargs = dict(
        pool=_CasBackend(conn),
        memory_engine=types.SimpleNamespace(embeddings=object()),
        bank_id="bank1",
        # The merge path builds a search_vector UPDATE clause from the text-search
        # config, so these must be present (production defaults: native/english).
        config=types.SimpleNamespace(
            consolidation_dedup_threshold=threshold,
            text_search_extension="native",
            text_search_extension_native_language="english",
        ),
        dedup_llm_config=llm,
        updated_id=_UPDATED_ID,
        updated_text="Uzbek content on YouTube is very rich and growing.",
        updated_emb_str="[0.1, 0.2, 0.3]",  # already embedded by _execute_update_action
        tags=["t1"],
    )
    return kwargs, store, conn


async def test_dedup_update_merge_folds_into_twin_and_deletes_updated() -> None:
    kwargs, store, conn = _update_ctx()
    llm = kwargs["dedup_llm_config"]
    llm.call.return_value = _DedupDecision(action="merge", text="Uzbek YouTube content is very rich and growing.")
    with (
        _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.98)], store),
        _patch_store(store),
    ):
        await _dedup_reconcile_update(**kwargs)
    llm.call.assert_awaited_once()
    # The fold-into-twin is a CAS fold gated on fresh revision tokens; it folds only the
    # updated row's LIVE sources (snapshotted fresh + filtered FOR SHARE).
    store.cas_fold_observation.assert_awaited_once()
    fold_kwargs = store.cas_fold_observation.await_args.kwargs
    assert fold_kwargs["merged_text"] == "Uzbek YouTube content is very rich and growing."  # merged text on the twin
    assert fold_kwargs["observation_id"] == _TWIN_ID  # survivor = the twin
    assert fold_kwargs["add_source_ids"] == [str(s) for s in store._updated_sources]  # live updated-row sources
    # Then the updated row is deleted via CAS + its observation_history is reclaimed.
    store.cas_delete_memory.assert_awaited_once()
    assert store.cas_delete_memory.await_args.kwargs["unit_id"] == _UPDATED_ID
    assert conn.execute.await_count == 1  # observation_history delete only (row delete went via CAS)
    history_delete_args = conn.execute.await_args_list[0].args
    assert history_delete_args[2] == uuid.UUID(_UPDATED_ID)  # its history is reclaimed too


async def test_dedup_update_keep_does_not_merge() -> None:
    kwargs, store, conn = _update_ctx()
    llm = kwargs["dedup_llm_config"]
    llm.call.return_value = _DedupDecision(action="keep", reason="different growth claim")
    with _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.98)], store):
        await _dedup_reconcile_update(**kwargs)
    llm.call.assert_awaited_once()
    store.cas_fold_observation.assert_not_called()  # kept distinct → no fold
    store.cas_delete_memory.assert_not_called()  # → no delete


async def test_dedup_update_excludes_self() -> None:
    # The probe surfaces the updated observation itself at 1.0; it must be excluded so we don't
    # "merge" a row into itself. With no other candidate, there is no twin → no LLM, no writes.
    kwargs, store, conn = _update_ctx()
    llm = kwargs["dedup_llm_config"]
    with _patch_probe([_obs("its own current text", 1.0, oid=_UPDATED_ID)], store):
        await _dedup_reconcile_update(**kwargs)
    llm.call.assert_not_called()
    store.cas_fold_observation.assert_not_called()
    store.cas_delete_memory.assert_not_called()


async def test_dedup_update_no_twin_above_threshold() -> None:
    kwargs, store, conn = _update_ctx(threshold=0.97)
    llm = kwargs["dedup_llm_config"]
    with _patch_probe([_obs("loosely related", 0.8)], store):
        await _dedup_reconcile_update(**kwargs)
    llm.call.assert_not_called()
    store.cas_fold_observation.assert_not_called()
    store.cas_delete_memory.assert_not_called()


# ── dedup activation gate (_dedup_active) ─────────────────────────────────────
#
# Enabled by default (threshold < 1.0), but skipped on Oracle because the merge path is
# Postgres-only — so the feature can ship on-by-default without breaking Oracle.


def _gate_cfg(threshold: float):
    return types.SimpleNamespace(consolidation_dedup_threshold=threshold)


def _patch_backend(name: str):
    return patch(
        "hindsight_api.engine.consolidation.consolidator.get_config",
        return_value=types.SimpleNamespace(database_backend=name),
    )


def test_dedup_active_enabled_on_postgres() -> None:
    with _patch_backend("postgresql"):
        assert _dedup_active(_gate_cfg(0.97)) is True


def test_dedup_active_disabled_when_threshold_is_one() -> None:
    with _patch_backend("postgresql"):
        assert _dedup_active(_gate_cfg(1.0)) is False


def test_dedup_active_skipped_on_oracle() -> None:
    # PG-only merge path → dedup is skipped on Oracle even with a sub-1.0 threshold.
    with _patch_backend("oracle"):
        assert _dedup_active(_gate_cfg(0.97)) is False


def test_dedup_active_none_config() -> None:
    assert _dedup_active(None) is False


# ── connection-release fold guards (RETURNING-gated, live-source re-filter) ────
#
# The embed/LLM adjudication runs with no connection held; the fold then re-checks source
# liveness inside a short transaction and is RETURNING-gated so a twin that vanished (or a
# source deleted) during the connection-free window can't drop a CREATE or fold a dead id.


async def test_dedup_create_twin_vanished_returns_none_so_caller_creates() -> None:
    # If the twin is deleted during the (connection-free) LLM window, snapshot_memories finds no
    # row; the helper must return None so the caller still CREATEs.
    kwargs, store, conn = _ctx()
    store.snapshot_memories.return_value = []  # twin vanished
    llm = kwargs["dedup_llm_config"]
    llm.call.return_value = _DedupDecision(action="merge", text="merged text")
    with (
        _patch_embed(),
        _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.99)], store),
        _patch_store(store),
    ):
        result = await _dedup_reconcile_create(**kwargs)
    assert result is None  # twin gone → don't drop the CREATE
    store.cas_fold_observation.assert_not_called()


async def test_dedup_create_fold_uses_only_live_new_sources() -> None:
    kwargs, store, conn = _ctx()
    live_source_id = uuid.uuid4()
    deleted_source_id = uuid.uuid4()
    kwargs["create_source_ids"] = [deleted_source_id, live_source_id]
    conn.live_rows = [{"id": live_source_id}]  # only live_source_id survives liveness re-check
    llm = kwargs["dedup_llm_config"]
    llm.call.return_value = _DedupDecision(action="merge", text="merged text")
    with (
        _patch_embed(),
        _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.99)], store),
        _patch_store(store),
    ):
        result = await _dedup_reconcile_create(**kwargs)
    assert result == _TWIN_ID
    store.cas_fold_observation.assert_awaited_once()
    assert store.cas_fold_observation.await_args.kwargs["add_source_ids"] == [str(live_source_id)]


async def test_dedup_create_all_new_sources_deleted_returns_none() -> None:
    kwargs, store, conn = _ctx()
    kwargs["create_source_ids"] = [uuid.uuid4(), uuid.uuid4()]
    conn.live_rows = []  # every source gone under liveness re-check
    llm = kwargs["dedup_llm_config"]
    llm.call.return_value = _DedupDecision(action="merge", text="merged text")
    with (
        _patch_embed(),
        _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.99)], store),
        _patch_store(store),
    ):
        result = await _dedup_reconcile_create(**kwargs)
    assert result is None
    store.cas_fold_observation.assert_not_called()


async def test_dedup_update_twin_vanished_does_not_delete_updated() -> None:
    # If the twin vanished mid-window (no snapshot), the updated row must NOT be deleted.
    kwargs, store, conn = _update_ctx()
    llm = kwargs["dedup_llm_config"]
    store.snapshot_memories.side_effect = None
    store.snapshot_memories.return_value = []

    async def _snap_vanish(conn=None, fq_table=None, bank_id=None, unit_ids=None):
        return [] if unit_ids == [_TWIN_ID] else [_snap(_UPDATED_ID)]

    store.snapshot_memories.side_effect = _snap_vanish
    llm.call.return_value = _DedupDecision(action="merge", text="merged text")
    with (
        _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.98)], store),
        _patch_store(store),
    ):
        await _dedup_reconcile_update(**kwargs)
    store.cas_fold_observation.assert_not_called()  # twin gone → no fold attempted
    store.cas_delete_memory.assert_not_called()  # and no delete


async def test_dedup_update_fold_uses_only_live_updated_sources() -> None:
    kwargs, store, conn = _update_ctx()
    live_source_id = uuid.uuid4()
    kwargs["updated_text"] = "merged text"
    llm = kwargs["dedup_llm_config"]
    llm.call.return_value = _DedupDecision(action="merge", text="merged text")
    # Only live_source_id survives the fresh liveness re-check (deleted one is dropped).
    with (
        _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.98)], store),
        _patch_store(store),
        patch(
            "hindsight_api.engine.consolidation.consolidator._filter_live_source_memories",
            AsyncMock(return_value=[live_source_id]),
        ),
    ):
        await _dedup_reconcile_update(**kwargs)
    store.cas_fold_observation.assert_called_once()
    assert store.cas_fold_observation.call_args.kwargs["add_source_ids"] == [str(live_source_id)]
    store.cas_delete_memory.assert_called_once()  # fold succeeded → updated row deleted


async def test_dedup_update_all_updated_sources_deleted_skips_fold_and_delete() -> None:
    kwargs, store, conn = _update_ctx()
    llm = kwargs["dedup_llm_config"]
    llm.call.return_value = _DedupDecision(action="merge", text="merged text")
    with (
        _patch_probe([_obs("Uzbek content on YouTube is described as very rich.", 0.98)], store),
        _patch_store(store),
        patch(
            "hindsight_api.engine.consolidation.consolidator._filter_live_source_memories",
            AsyncMock(return_value=[]),  # every updated-row source gone
        ),
    ):
        await _dedup_reconcile_update(**kwargs)
    store.cas_fold_observation.assert_not_called()
    store.cas_delete_memory.assert_not_called()


# ── _process_memory_batch create-contract (created vs skipped) ────────────────


def _batch_engine():
    return types.SimpleNamespace(_consolidation_llm_config=types.SimpleNamespace(with_config=lambda *a, **k: object()))


async def _run_create_batch(create_action_result: str):
    from hindsight_api.engine.consolidation import consolidator as C

    mem_id = str(uuid.uuid4())
    memories = [{"id": mem_id, "text": "Uzbek YouTube content is very rich.", "tags": []}]
    create = C._CreateAction(text="Uzbek YouTube content is very rich.", source_fact_ids=[mem_id])
    llm_result = C._BatchLLMResult(creates=[create])
    with (
        patch.object(
            C,
            "_find_related_observations",
            new=AsyncMock(return_value=types.SimpleNamespace(results=[], source_facts={})),
        ),
        patch.object(C, "_consolidate_batch_with_llm", new=AsyncMock(return_value=llm_result)),
        patch.object(C, "_effective_scope_limit", return_value=-1),
        patch.object(C, "_dedup_active", return_value=True),
        patch.object(C, "_dedup_reconcile_create", new=AsyncMock(return_value=None)),
        patch.object(C, "_execute_create_action", new=AsyncMock(return_value=create_action_result)) as create_action,
    ):
        result = await C._process_memory_batch(
            pool=object(),
            memory_engine=_batch_engine(),
            llm_config=object(),
            bank_id="bank1",
            memories=memories,
            request_context=object(),
            config=object(),
        )
    return result, create_action, mem_id


async def test_process_batch_creates_when_dedup_target_vanished() -> None:
    # Caller contract: when _dedup_reconcile_create returns None (twin vanished mid-window),
    # _process_memory_batch must still CREATE the observation instead of dropping it.
    result, create_action, mem_id = await _run_create_batch("created")
    create_action.assert_awaited_once()
    assert create_action.await_args.kwargs["text"] == "Uzbek YouTube content is very rich."
    assert create_action.await_args.kwargs["source_memory_ids"] == [mem_id]
    assert result == ([{"action": "created"}], 0, False)


async def test_process_batch_reports_skipped_when_create_skipped() -> None:
    # _execute_create_action returns "skipped" (all sources deleted in the write txn) ->
    # _process_memory_batch must NOT mark the memory created; it falls through to skipped.
    result, _create_action, _mem_id = await _run_create_batch("skipped")
    assert result == ([{"action": "skipped", "reason": "no_durable_knowledge"}], 0, False)


async def test_process_batch_reports_created_when_create_created() -> None:
    # _execute_create_action returns "created" -> the memory is marked created.
    result, _create_action, _mem_id = await _run_create_batch("created")
    assert result == ([{"action": "created"}], 0, False)
