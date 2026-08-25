"""Consolidation engine for automatic observation creation from memories.

The consolidation engine runs as a background job after retain operations complete.
It processes new memories and either:
- Creates new observations from novel facts
- Updates existing observations when new evidence supports/contradicts/refines them

Observations are stored in memory_units with fact_type='observation' and include:
- proof_count: Number of supporting memories
- source_memory_ids: Array of memory UUIDs that contribute to this observation
- history: JSONB tracking changes over time

NOTE: Observations are distinct from mental models (pinned reflections).
- Observations: auto-generated bottom-up by this engine from raw facts (memory_units table, fact_type='observation')
- Mental models: user-defined queries stored in the mental_models table, refreshed on demand via reflect
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from itertools import combinations
from typing import TYPE_CHECKING, Any, Literal

import asyncpg
from pydantic import BaseModel, field_validator

from ...config import get_config
from ...worker.stage import set_stage
from ..db import DatabaseBackend
from ..db_utils import acquire_with_retry
from ..llm_trace import (
    record_created_memory_ids,
    record_source_memory_ids,
    reset_trace_context,
    set_trace_context,
    trace_context_of,
)
from ..llm_wrapper import sanitize_llm_output
from ..memories import (
    CASOutcome,
    FactRecord,
    MemoryPatch,
    MemorySnapshot,
    StoredMemory,
    get_memories,
)
from ..memory_engine import Budget, fq_table
from ..retain import embedding_utils
from .prompts import (
    build_consolidation_input,
    build_consolidation_system_prompt,
)

if TYPE_CHECKING:
    from asyncpg import Connection

    from ...api.http import RequestContext
    from ..memory_engine import MemoryEngine
    from ..response_models import MemoryFact, RecallResult

logger = logging.getLogger(__name__)


class UnsupportedConsolidationDialectError(RuntimeError):
    """Store-CAS consolidation v1 is unavailable for this database dialect."""


class _BatchStaleError(Exception):
    """Raised inside the Phase-B transaction when a prepared plan is stale under the guard.

    Ruling 1: a stale plan aborts the ENTIRE Phase-B attempt. Raising this INSIDE the
    ``async with conn.transaction()`` scope makes the context manager roll back every write
    made by earlier plans; it is caught ONLY outside that scope so the rollback is guaranteed.
    The carried ``reason`` is logged for diagnostics and the batch contributes zero progress.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@asynccontextmanager
async def _write_group(pool, conn=None):
    """Yield a connection for one logical write-group.

    Serial path (``conn is None``): acquire a short-lived connection from the pool and open
    one transaction — the executor owns both, so each action commits independently.

    Phase-B path (``conn`` given): yield the caller-owned connection **without opening a
    transaction on it** — the caller already holds the bank-level write-group, and nesting
    a transaction here would both violate the one-write-group invariant and fail inside
    asyncpg ("cannot perform this operation inside a transaction"). All observation writes
    and source marks then share that single connection/transaction fate.
    """
    if conn is not None:
        yield conn
        return
    async with acquire_with_retry(pool) as c:
        async with c.transaction():
            yield c


async def _gather_or_cancel(coros: list[Any]) -> list[Any]:
    """``asyncio.gather`` that leaves no task running behind it.

    Plain ``asyncio.gather`` re-raises the first exception immediately but does
    NOT cancel its siblings — they keep running detached. In consolidation that
    is actively harmful: the failure propagates out of ``run_consolidation_job``
    to the worker, which marks the operation failed and re-queues it with a 5s
    base backoff, while the orphaned tag groups are still calling the LLM,
    stamping ``mark_consolidated`` and committing write-groups. The per-scope
    ``scope_locks`` are local to one dispatch, so nothing serialises an orphan
    against the retry, and the "batches within a group run serially" invariant
    that keeps two consolidators out of the same observation scope is broken
    exactly when it matters.

    So: cancel the outstanding tasks and await them before propagating. A
    cancelled batch's writes stay invisible (its witness row is never
    committed) and are resolved by the recovery sweep, which is the same state
    a crash would leave.

    Deliberately not ``asyncio.TaskGroup``: it wraps failures in an
    ``ExceptionGroup``, and the worker's ``_is_non_retryable_task_error`` does
    ``isinstance`` checks on the raised exception — a wrapped
    ``IntegrityConstraintViolationError`` would be misclassified as retryable
    and retried forever. This helper re-raises the original exception unchanged.
    """
    tasks = [asyncio.ensure_future(c) for c in coros]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:
            if not t.done():
                t.cancel()
        # Await the cancellations before propagating: returning while they are
        # still unwinding would reintroduce the very overlap this prevents.
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _native_search_vector_update(config, param: str) -> str:
    """UPDATE-clause fragment that repopulates ``search_vector`` inline, or ''
    when the backend does not maintain a native tsvector column that way.

    ``to_tsvector(...)::regconfig`` is PostgreSQL-only. On Oracle ``search_vector``
    is a CLOB maintained by Oracle's own text index rather than an inline
    tsvector, so emit nothing there (mirrors the insert path, which gates
    ``search_vector`` on the PG-only ``pg_search_vector_expr``). Without this
    guard the PG expression reaches Oracle and fails with DPY-4010 (the
    ``::regconfig`` cast becomes an unbound ``:REGCONFIG`` placeholder).
    """
    from ..schema import _is_oracle  # noqa: PLC0415

    if config.text_search_extension != "native" or _is_oracle():
        return ""
    lang = config.text_search_extension_native_language
    return f",\n            search_vector = to_tsvector('{lang}'::regconfig, COALESCE({param}, ''))"


def _norm_obs_text(text: str) -> str:
    """Whitespace-normalised observation text for exact-duplicate matching.

    Collapses runs of whitespace only; case is preserved. The reconciliation guard
    drops a CREATE on the premise that an exact-text match loses no information — but
    case-folding would also drop a create differing only in case (e.g. "TLS" vs "tls"),
    which *does* lose information, so we match case-sensitively.
    """
    return " ".join((text or "").split()).strip()


def _duplicate_create_target(
    create_text: str,
    shown_obs_by_text: "dict[str, MemoryFact]",
    update_texts: set[str],
) -> str | None:
    """Return a human label for what ``create_text`` duplicates, or None if novel.

    A CREATE is a duplicate when its normalised text matches an observation that was
    already shown to the LLM, or the text of an UPDATE issued in the same response
    (the model occasionally UPDATEs the twin to text X and also CREATEs X). Exact-text
    match means no information is lost by dropping the CREATE.
    """
    norm = _norm_obs_text(create_text)
    matched = shown_obs_by_text.get(norm)
    if matched is not None:
        return f"shown observation {str(matched.id)[:8]}"
    if norm in update_texts:
        return "an UPDATE in this response"
    return None


# Top-K existing observations probed (by the new observation's own embedding) when
# semantic dedup is enabled. Small: we only need the nearest few candidates.
_DEDUP_TOP_K = 5


class _DedupDecision(BaseModel):
    """Focused 1-by-1 verdict for whether a new observation duplicates an existing one."""

    action: Literal["merge", "keep"] = "keep"
    text: str = ""  # the synthesized merged observation (when action == "merge")
    reason: str = ""

    @field_validator("action", mode="before")
    @classmethod
    def _normalize_action(cls, value: object) -> str:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"merge", "keep"}:
                return normalized

        logger.warning("Invalid consolidation dedup action %r; defaulting to keep", value)
        return "keep"


def _dedup_decision_from_response(raw: Any) -> _DedupDecision:
    try:
        if isinstance(raw, _DedupDecision):
            return raw
        if isinstance(raw, str):
            return _DedupDecision.model_validate_json(raw)
        return _DedupDecision.model_validate(raw)
    except ValueError as exc:
        logger.warning("Invalid consolidation dedup response %r; defaulting to keep: %s", raw, exc)
        return _DedupDecision(action="keep", reason="invalid structured response")


_DEDUP_PROMPT = """You reconcile long-term memory observations. A NEW observation is about to be \
stored, and it is highly similar to an EXISTING one:

[NEW] {new}
[EXISTING] {existing}

Respond with ONLY one valid JSON object matching one of these shapes:

For duplicate facts:
{{"action": "merge", "text": "...", "reason": "..."}}

For distinct facts:
{{"action": "keep", "text": "", "reason": "..."}}

Do NOT use key=value lines, markdown fences, or any text outside the JSON object.

If they assert the SAME fact (wording aside), set "action" to "merge" and provide "text": a \
single observation that preserves EVERY detail from both. If they differ in ANY important detail \
— a number/quantity, a named entity or language, a negation, or a condition — set "action" to \
"keep" and "text" to an empty string."""


def _dedup_active(config: Any) -> bool:
    """Whether create/update semantic dedup runs for this consolidation.

    Enabled when the resolved threshold is < 1.0, EXCEPT on Oracle: the merge path uses
    Postgres-only SQL (``unnest``/``array_agg``, ``UPDATE ... FROM``), so on Oracle dedup is
    skipped — it behaves exactly as it did before this feature, regardless of the configured
    threshold. This is why the feature can ship enabled-by-default without breaking Oracle.
    """
    if config is None or getattr(config, "consolidation_dedup_threshold", 1.0) >= 1.0:
        return False
    return get_config().database_backend != "oracle"


@dataclass
class _DedupOutcome:
    """Result of probing one observation against its in-scope neighbours.

    ``best_id`` is the nearest observation at/above the threshold (None if none),
    ``merged_text`` is the LLM-synthesized union text (set only when ``should_merge``).
    ``candidate_ids`` records EVERY observation id probed during the Phase-A
    adjudication (the bounded top-K, in scope) — the Phase-A candidate snapshot.
    Ruling 2 re-checks this snapshot under the bank guard: if a fresh in-scope
    candidate at/above threshold appears that was NOT probed here, the CREATE is
    stale (a semantic twin was introduced during the LLM window).
    """

    best_id: str | None
    merged_text: str
    should_merge: bool
    # The twin's text at probe time. Guards the fold against a concurrent survivor
    # rewrite during the connection-free LLM window (set on the two non-None returns).
    best_text: str = ""
    # Phase-A candidate snapshot (all probed ids, not just best).
    candidate_ids: set[str] = field(default_factory=set)
    # Opaque CAS token of every probed candidate, keyed by unit id. Tokens come from
    # the StoredMemory snapshot used for adjudication, not a later re-read.
    candidate_revisions: dict[str, str] = field(default_factory=dict)


async def _dedup_adjudicate(
    pool: DatabaseBackend,
    memory_engine: "MemoryEngine",
    bank_id: str,
    config: Any,
    dedup_llm_config: Any,
    anchor_text: str,
    anchor_emb_str: str | None,
    tags: list[str] | None,
    exclude_id: str | None,
) -> _DedupOutcome:
    """Probe one observation's embedding against in-scope observations and adjudicate a merge.

    Anchored on the observation text — the correct obs<->obs comparison, unlike consolidation
    recall which is anchored on the raw fact. Returns the nearest observation at/above
    ``consolidation_dedup_threshold`` and, when found, the LLM's focused 1-by-1 merge-or-keep
    verdict (scope ``consolidation_dedup``): the LLM reads both texts, so a word-level difference
    (number / negation / entity) is respected. ``exclude_id`` skips the anchor observation itself
    (used by the UPDATE path, where the anchor row already exists and would self-match at 1.0).
    ``anchor_emb_str`` reuses an already-computed embedding (the UPDATE path just embedded it);
    pass None to embed ``anchor_text`` here (the CREATE path).

    The embedder and the LLM both run with NO connection held; only the semantic+BM25 probe
    briefly borrows a short-lived connection.
    """
    from ..memories import get_memories

    threshold = config.consolidation_dedup_threshold
    if anchor_emb_str is None:
        embs = await embedding_utils.generate_embeddings_batch(memory_engine.embeddings, [anchor_text])
        if not embs:
            return _DedupOutcome(best_id=None, merged_text="", should_merge=False)
        anchor_emb_str = str(embs[0])
    tags_match = "all_strict" if tags else "any"
    # Dedup only needs the dense/keyword arms over observations — no graph, no temporal window.
    grouped = await get_memories().recall_unified(
        conn=pool,
        bank_id=bank_id,
        fact_types=["observation"],
        query_embedding=anchor_emb_str,
        query_text=anchor_text,
        limit=_DEDUP_TOP_K,
        tags=tags,
        tags_match=tags_match,
        enable_graph=False,
        temporal_window=None,
    )
    results = grouped["observation"].semantic
    # Ruling 2: capture the Phase-A candidate snapshot — every observation id probed
    # (the bounded in-scope top-K), so the under-guard re-check can detect a fresh
    # semantic twin introduced during the LLM window.
    probed_ids = [str(r.id) for r in results]
    snaps = await _snapshot_observations(pool, bank_id, probed_ids)
    candidate_ids: set[str] = set(snaps)
    candidate_revisions: dict[str, str] = {uid: snap.revision for uid, snap in snaps.items()}
    # Adjudicate against snapshot-backed text so the focused LLM and the stored token
    # describe the same StoredMemory (judge 76f0ac68).
    best_id: str | None = None
    best_text = ""
    best_sim = threshold  # only candidates at/above the threshold are considered
    for r in results:
        rid = str(r.id)
        snap = snaps.get(rid)
        if snap is None:
            continue  # vanished between probe and snapshot; drop before adjudication
        if exclude_id is not None and rid == exclude_id:
            continue  # never match the anchor observation against itself
        sim = r.similarity or 0.0
        if sim >= best_sim:
            best_id, best_text, best_sim = rid, snap.memory.text, sim

    if best_id is None:
        return _DedupOutcome(
            best_id=None,
            merged_text="",
            should_merge=False,
            candidate_ids=candidate_ids,
            candidate_revisions=candidate_revisions,
        )

    decision = _dedup_decision_from_response(
        await dedup_llm_config.call(
            messages=[{"role": "user", "content": _DEDUP_PROMPT.format(new=anchor_text, existing=best_text)}],
            response_format=_DedupDecision,
            scope="consolidation_dedup",
            strict_schema=get_config().llm_strict_schema_consolidation,
        )
    )
    if decision.action != "merge":
        return _DedupOutcome(
            best_id=best_id,
            merged_text="",
            should_merge=False,
            best_text=best_text,
            candidate_ids=candidate_ids,
            candidate_revisions=candidate_revisions,
        )
    merged_text = (sanitize_llm_output(decision.text) or "").strip() or best_text
    return _DedupOutcome(
        best_id=best_id,
        merged_text=merged_text,
        should_merge=True,
        best_text=best_text,
        candidate_ids=candidate_ids,
        candidate_revisions=candidate_revisions,
    )


async def _dedup_reconcile_create(
    pool: DatabaseBackend,
    memory_engine: "MemoryEngine",
    bank_id: str,
    config: Any,
    dedup_llm_config: Any,
    create_text: str,
    create_source_ids: list[uuid.UUID],
    tags: list[str] | None,
    txn=None,
    *,
    conn=None,
    outcome=None,
    expected_revision: str | None = None,
) -> str | None:
    """Semantic dedup for a single CREATE (create-time, focused 1-by-1).

    On "merge", folds the new source facts + the synthesized text into the existing
    observation and returns its id (caller skips the CREATE). Returns None when there is
    no near twin or the LLM keeps them distinct.

    Split for the Phase A / Phase B design (design §4.4): the probe/embed/LLM adjudication
    runs connection-free in Phase A; the CAS-protected fold is ``_dedup_fold_create`` and runs
    on the caller-owned Phase-B connection when ``conn`` is given (or on a short-lived
    connection+transaction on the serial path). Pass a pre-computed ``outcome`` to skip
    re-adjudication when it already ran in Phase A.
    """
    if outcome is None:
        outcome = await _dedup_adjudicate(
            pool, memory_engine, bank_id, config, dedup_llm_config, create_text, None, tags, exclude_id=None
        )
        if not outcome.should_merge or outcome.best_id is None:
            return None

    store = get_memories()
    async with _write_group(pool, conn) as conn:
        return await _dedup_fold_create(
            store=store,
            conn=conn,
            memory_engine=memory_engine,
            bank_id=bank_id,
            config=config,
            outcome=outcome,
            create_source_ids=create_source_ids,
            txn=txn,
            expected_revision=expected_revision,
        )


async def _dedup_fold_create(
    *,
    store,
    conn,
    memory_engine: "MemoryEngine",
    bank_id: str,
    config: Any,
    outcome,
    create_source_ids: list[uuid.UUID],
    txn=None,
    expected_revision: str | None = None,
) -> str | None:
    """CAS-protected Phase-B fold of a CREATE's sources into its near-twin (design §4.4).

    Runs inside the caller's write-group transaction (the bank guard when in Phase B).
    Re-checks source liveness fresh, then folds via the store CAS seam: SQL stores use
    ``cas_fold_observation`` gated on a freshly-snapshotted revision token (strictly stronger
    than the old RETURNING text-guard — it catches any authoritative-field change, not just a
    text rewrite); non-SQL stores fall back to ``_reconcile_merge_via_store`` until they ship a
    real CAS implementation. Returns the twin id on APPLIED; None when there is no fold so the
    caller proceeds with the CREATE (nothing is lost).
    """
    # Re-check liveness inside the fold transaction; CREATE performed the slow embed/LLM
    # work off-connection, so sources may have been deleted since the decision was made.
    live_source_ids = await _filter_live_source_memories(conn, bank_id, create_source_ids)
    if not live_source_ids:
        return None

    if store.writes_memory_rows_in_sql_for(bank_id):
        expected_rev = expected_revision or ((outcome.candidate_revisions or {}).get(str(outcome.best_id)))
        if not expected_rev:
            logger.debug(
                "[CONSOLIDATION] dedup-merge target %s has no Phase-A revision; proceeding with CREATE",
                outcome.best_id[:8],
            )
            return None
        folded = await store.cas_fold_observation(
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            observation_id=outcome.best_id,
            expected_revision=expected_rev,
            merged_text=outcome.merged_text,
            add_source_ids=[str(s) for s in live_source_ids],
        )
        if folded != CASOutcome.APPLIED:
            # Twin changed since adjudication — do not clobber it; proceed with CREATE.
            logger.debug(
                "[CONSOLIDATION] dedup-merge target %s changed during window; proceeding with CREATE",
                outcome.best_id[:8],
            )
            return None
        return outcome.best_id

    await _reconcile_merge_via_store(
        store, conn, memory_engine, bank_id, outcome.best_id, outcome.merged_text, live_source_ids, txn=txn
    )
    return outcome.best_id


async def _dedup_reconcile_update(
    pool: DatabaseBackend,
    memory_engine: "MemoryEngine",
    bank_id: str,
    config: Any,
    dedup_llm_config: Any,
    updated_id: str,
    updated_text: str,
    updated_emb_str: str | None,
    tags: list[str] | None,
    txn=None,
    *,
    conn=None,
    outcome=None,
) -> None:
    """Semantic dedup for an UPDATE (after the observation was rewritten + re-embedded).

    An UPDATE rewrites an observation's text and re-embeds it, so its vector can drift to
    within threshold of a DIFFERENT existing observation. The create-time guard never sees
    this (it only runs on CREATE), so without this the two persist as a near-duplicate pair —
    the measured residual-duplicate source. Probe the updated observation's new embedding
    against the others (excluding itself); on "merge", fold the just-updated observation's
    sources into the twin, persist the merged text, and DELETE the updated row. Unlike the
    CREATE path the row already exists, so reconciliation is a fold-and-delete, not a skip.

    Split for the Phase A / Phase B design (design §4.4): adjudication runs connection-free in
    Phase A; ``_dedup_fold_update`` runs the CAS-protected fold-and-delete on the caller-owned
    Phase-B connection when ``conn`` is given (or on a short-lived connection+transaction on
    the serial path). Pass a pre-computed ``outcome`` to skip re-adjudication when it already
    ran in Phase A.
    """
    if outcome is None:
        outcome = await _dedup_adjudicate(
            pool,
            memory_engine,
            bank_id,
            config,
            dedup_llm_config,
            updated_text,
            updated_emb_str,
            tags,
            exclude_id=updated_id,
        )
        if not outcome.should_merge or outcome.best_id is None:
            return

    store = get_memories()
    async with _write_group(pool, conn) as conn:
        await _dedup_fold_update(
            store=store,
            conn=conn,
            memory_engine=memory_engine,
            bank_id=bank_id,
            config=config,
            outcome=outcome,
            updated_id=updated_id,
            updated_text=updated_text,
            txn=txn,
        )


async def _dedup_fold_update(
    *,
    store,
    conn,
    memory_engine: "MemoryEngine",
    bank_id: str,
    config: Any,
    outcome,
    updated_id: str,
    updated_text: str,
    txn=None,
) -> None:
    """CAS-protected Phase-B fold-and-delete of an updated observation into its twin.

    Runs inside the caller's write-group transaction (the bank guard when in Phase B).

    The all_strict/any tag match guarantees twin and updated share scope, so dropping the
    updated row's tags loses no visibility. Temporal fields follow the surviving twin (minimal
    scope; matches create). The fold + delete share one logical write-group so the twin gains
    the sources exactly as the redundant row is removed; adjudication already ran connection-free.

    SQL stores fold via ``cas_fold_observation`` gated on fresh revision tokens for both rows
    (strictly stronger than the old RETURNING text-guard) and delete via ``cas_delete_memory``;
    non-SQL stores fall back to ``_reconcile_merge_via_store`` until they ship real CAS.
    """
    if store.writes_memory_rows_in_sql_for(bank_id):
        # Snapshot both rows fresh under the caller's transaction. Lock order must be
        # sources-before-observation: _filter_live_source_memories takes FOR SHARE on SOURCE rows
        # first, then snapshot_memories/cas take FOR UPDATE on observation rows — same order as the
        # normal write paths (_create_observation_directly / _execute_update_action).
        upd_snap = await store.snapshot_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[updated_id])
        if not upd_snap:
            # Updated row vanished during the LLM window — nothing to reconcile.
            return
        upd_sources = list(upd_snap[0].memory.source_memory_ids or [])
        live_u_sources = await _filter_live_source_memories(conn, bank_id, upd_sources)
        if not live_u_sources:
            return

        twin_snap = await store.snapshot_memories(
            conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[outcome.best_id]
        )
        if not twin_snap:
            # Twin vanished during the LLM window — keep the updated row as a distinct observation.
            return

        folded = await store.cas_fold_observation(
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            observation_id=outcome.best_id,
            expected_revision=twin_snap[0].revision,
            merged_text=outcome.merged_text,
            add_source_ids=[str(s) for s in live_u_sources],
        )
        if folded != CASOutcome.APPLIED:
            # Twin changed during the window — keep the updated row instead of folding a stale twin.
            return
        # Fold applied: delete the now-redundant updated row via CAS + its history.
        await store.cas_delete_memory(
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            unit_id=updated_id,
            expected_revision=upd_snap[0].revision,
        )
        await _delete_observation_history(conn, bank_id, updated_id)
        logger.info(
            "[CONSOLIDATION] dedup-merged updated observation %s into %s (cosine>=%.2f)",
            updated_id[:8],
            outcome.best_id[:8],
            config.consolidation_dedup_threshold,
        )
        return

    updated_obs = await store.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[updated_id])
    updated_sources = list(updated_obs[0].source_memory_ids or []) if updated_obs else []
    live_u_sources = await _filter_live_source_memories(conn, bank_id, updated_sources)
    if not live_u_sources:
        return
    await _reconcile_merge_via_store(
        store, conn, memory_engine, bank_id, outcome.best_id, outcome.merged_text, live_u_sources, txn=txn
    )
    await _execute_delete_action(conn, bank_id, updated_id, txn=txn)


@dataclass
class _BatchDeltas:
    """Per-LLM-batch deltas, merged into the job's running stats after dispatch.

    Returned by value rather than mutated into the outer ``stats`` /
    ``consolidated_tags`` so parallel batches cannot race on those shared
    structures (the merge happens once, serially, after dispatch completes).
    """

    stats: dict[str, int]
    tags: set[str]
    cancelled: bool
    # Ruling 3: memory ids whose batch exhausted its stale-reprepare budget this
    # invocation. The outer loop must NOT immediately re-fetch them in the same job
    # (no busy loop); they stay unconsolidated+unfailed for a LATER job invocation.
    retry_exhausted_ids: set[str] = field(default_factory=set)


def _parse_observation_scopes(memory: dict[str, Any]) -> Any:
    """Parse the per-memory ``observation_scopes`` value.

    The value arrives already decoded when read through the memories store (its
    reader coerces the JSONB column) or as raw JSON text from a driver without a
    JSONB codec. A scalar mode such as ``"per_tag"`` decodes to a bare string that
    is not itself valid JSON, so a blind ``json.loads`` would raise on it — try to
    parse, but treat an unparseable string as an already-decoded scalar.
    """
    raw = memory.get("observation_scopes")
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _resolve_obs_tags_list(memory: dict[str, Any]) -> list[list[str]] | None:
    """Resolve a memory's ``observation_scopes`` spec into concrete scope tags.

    Returns ``None`` for the default ``combined``-mode single pass (caller uses
    the memory's own tags). Returns a list[list[str]] when the memory requested
    multi-pass scoping (``per_tag``, ``all_combinations``, ``shared``, or an
    explicit list).

    ``shared`` resolves to ``[[]]`` — a single pass over the empty (untagged)
    scope. The created observation carries no tags and recall/dedup match it with
    ``tags_match="any"``, so every memory consolidates into one shared observation
    regardless of its own tags. Use it to deduplicate across volatile per-call
    provenance tags (e.g. per-session ids) without dropping those tags from the
    source facts.
    """
    parsed = _parse_observation_scopes(memory)
    tags = list(memory.get("tags") or [])

    if parsed == "per_tag":
        return [[t] for t in tags] if tags else None
    if parsed == "all_combinations":
        if not tags:
            return None
        return [list(c) for r in range(1, len(tags) + 1) for c in combinations(tags, r)]
    if parsed == "shared":
        return [[]]
    if parsed == "combined" or parsed is None:
        return None
    return parsed  # explicit list[list[str]]


def _resolve_write_scopes(memory: dict[str, Any]) -> list[frozenset[str]]:
    """Return the observation scopes a memory will write to, as frozensets.

    Used by the parallel dispatcher to acquire one lock per scope before
    processing a tag group, so that two groups whose write-scope sets overlap
    serialise on the overlapping scopes rather than racing on the same
    observation row. The mapping mirrors ``_resolve_obs_tags_list`` exactly:

    - ``combined`` / ``None``    -> ``[frozenset(memory.tags)]``
    - ``per_tag``                -> ``[frozenset({t}) for t in memory.tags]``
    - ``all_combinations``       -> one frozenset per nonempty subset of tags
    - ``shared``                 -> ``[frozenset()]`` (the single untagged scope)
    - explicit ``list[list[str]]`` -> one frozenset per declared scope

    Empty-tag memories collapse to a single ``frozenset()`` in all modes so they
    still take exactly one lock and serialise against other untagged work.
    """
    parsed = _parse_observation_scopes(memory)
    tags = list(memory.get("tags") or [])

    if parsed == "per_tag":
        return [frozenset([t]) for t in tags] if tags else [frozenset()]
    if parsed == "all_combinations":
        if not tags:
            return [frozenset()]
        return [frozenset(c) for r in range(1, len(tags) + 1) for c in combinations(tags, r)]
    if parsed == "shared":
        return [frozenset()]
    if parsed == "combined" or parsed is None:
        return [frozenset(tags)]
    return [frozenset(s) for s in parsed]  # explicit list[list[str]]


def _scope_sort_key(scope: frozenset[str]) -> tuple[str, ...]:
    """Total ordering on scope frozensets for deadlock-free lock acquisition.

    Every parallel group acquires its scope locks in this same order, so two
    groups that share any subset of scopes cannot acquire them in opposite
    orders and deadlock.
    """
    return tuple(sorted(scope))


def _source_fingerprint(m: dict[str, Any]) -> dict[str, Any]:
    """Capture the mutation-relevant authoritative state of a source memory at Phase-A time.

    Phase B (design §4.2 step 3-4, §6.1) re-locks the source rows under the bank guard
    and compares against this snapshot. Any change to a consolidation-relevant field
    (text, tags, temporal fields) invalidates the prepared CREATE/UPDATE for that source.

    Only keys PRESENT on ``m`` are included: ``_fetch_unconsolidated_rows`` returns a subset
    of columns (no proof_count/source_memory_ids/consolidated_at), so those are simply not
    compared here — a missing key means "we did not observe it in Phase A" and cannot be
    treated as a change. The authoritative ``consolidated_at`` consumed check is done as an
    explicit SQL predicate in :func:`_fresh_source_validation`, not via this fingerprint.
    """
    out: dict[str, Any] = {}
    if "text" in m:
        out["text"] = m.get("text")
    if "fact_type" in m:
        out["fact_type"] = m.get("fact_type")
    if "tags" in m:
        out["tags"] = sorted(m.get("tags") or [])
    if "event_date" in m:
        out["event_date"] = _iso_or_none(m.get("event_date"))
    if "occurred_start" in m:
        out["occurred_start"] = _iso_or_none(m.get("occurred_start"))
    if "occurred_end" in m:
        out["occurred_end"] = _iso_or_none(m.get("occurred_end"))
    if "mentioned_at" in m:
        out["mentioned_at"] = _iso_or_none(m.get("mentioned_at"))
    return out


def _iso_or_none(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    return str(v)


async def _filter_live_source_memories(
    conn: "Connection",
    bank_id: str,
    source_memory_ids: list[uuid.UUID],
) -> list[uuid.UUID]:
    """Return only the source memory ids that still exist in the bank.

    The SQL store takes a ``FOR SHARE`` lock on the surviving rows so a concurrent
    delete can't remove one between this check and the observation write that
    follows — without it, a source deleted in that window would leave an orphan
    observation until the next sweep, because the delete path's stale-observation
    sweep only catches observations that already exist when it runs. (Oracle has no
    ``FOR SHARE``; the SQL rewriter promotes it to ``FOR UPDATE`` — more
    conservative, still correct.) A store that keeps memories outside SQL has its
    own concurrency model, so it answers with an unlocked existence check.
    """
    if not source_memory_ids:
        return []
    store = get_memories()
    if store.writes_memory_rows_in_sql_for(bank_id):
        rows = await conn.fetch(
            f"SELECT id FROM {fq_table('memory_units')} WHERE id = ANY($1::uuid[]) AND bank_id = $2 FOR SHARE",
            source_memory_ids,
            bank_id,
        )
        live = {str(r["id"]) for r in rows}
    else:
        present = await store.get_memories(
            conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[str(mid) for mid in source_memory_ids]
        )
        live = {str(m.unit_id) for m in present}
    return [mid for mid in source_memory_ids if str(mid) in live]


async def _any_live_source_memory(
    conn: "Connection",
    bank_id: str,
    source_memory_ids: list[uuid.UUID],
) -> bool:
    """Cheap, non-locking existence check used as a preflight before embedding.

    Lets the create/update executors skip the (slow) embedder when every source
    memory is already gone, restoring the pre-refactor short-circuit. The
    authoritative, FOR SHARE liveness check still runs inside the write txn.
    """
    if not source_memory_ids:
        return False
    store = get_memories()
    if store.writes_memory_rows_in_sql_for(bank_id):
        found = await conn.fetchval(
            f"SELECT 1 FROM {fq_table('memory_units')} WHERE id = ANY($1::uuid[]) AND bank_id = $2 LIMIT 1",
            source_memory_ids,
            bank_id,
        )
        return found is not None
    present = await store.get_memories(
        conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[str(mid) for mid in source_memory_ids]
    )
    return bool(present)


def _fingerprint_matches(expected: dict[str, Any], fresh: dict[str, Any]) -> bool:
    """Compare a Phase-A source fingerprint against a fresh store snapshot field set.

    Both are produced by :func:`_source_fingerprint`, so missing keys compare as
    None (absent on both sides). The consolidated marker is compared too: a source
    consumed by a concurrent writer differs in ``consolidated_at`` and invalidates
    the prepared plan.
    """
    return all(expected.get(k) == fresh.get(k) for k in expected)


async def _fresh_source_validation(
    conn: "Connection",
    bank_id: str,
    source_ids: list[uuid.UUID],
    expected_fingerprints: dict[str, dict[str, Any]],
) -> str:
    """Fresh source-ID + candidate-set revalidation under the Phase-B bank guard.

    Design §4.2 step 3-4 and §6.1: after the bank row is ``FOR UPDATE``-locked, re-lock
    the source rows in stable ID order (bank -> source lock order) and validate that each
    source (a) still exists, (b) is still unconsolidated, and (c) is unchanged since its
    Phase-A snapshot. Also revalidate the candidate set: if an observation already exists
    that references any of these sources (a twin created by a concurrent writer during the
    LLM window), the CREATE is stale.

    Returns ``"ok"`` when everything matches, or a short stale reason describing why the
    prepared write must be dropped with zero writes.
    """
    if not source_ids:
        return "no_sources"
    store = get_memories()
    ordered = sorted(source_ids, key=lambda uid: str(uid))
    if store.writes_memory_rows_in_sql_for(bank_id):
        # Lock + read the authoritative state in one stable-order statement.
        rows = await conn.fetch(
            f"SELECT id, text, fact_type, tags, event_date, occurred_start, occurred_end,"
            f" mentioned_at, proof_count, source_memory_ids, consolidated_at"
            f" FROM {fq_table('memory_units')}"
            f" WHERE bank_id = $1 AND id = ANY($2::uuid[]) ORDER BY id FOR UPDATE",
            bank_id,
            ordered,
        )
        fresh_by_id: dict[str, dict[str, Any]] = {}
        fresh_consumed: dict[str, bool] = {}
        for r in rows:
            fresh_by_id[str(r["id"])] = _source_fingerprint(
                {
                    "text": r["text"],
                    "fact_type": r["fact_type"],
                    "tags": r["tags"] or [],
                    "event_date": r["event_date"],
                    "occurred_start": r["occurred_start"],
                    "occurred_end": r["occurred_end"],
                    "mentioned_at": r["mentioned_at"],
                }
            )
            fresh_consumed[str(r["id"])] = r["consolidated_at"] is not None
        for sid in ordered:
            sid_str = str(sid)
            fresh = fresh_by_id.get(sid_str)
            if fresh is None:
                return f"source_deleted:{sid_str}"
            if fresh_consumed.get(sid_str):
                return f"source_consumed:{sid_str}"
            expected = expected_fingerprints.get(sid_str)
            if expected is not None and not _fingerprint_matches(expected, fresh):
                return f"source_changed:{sid_str}"
        # Candidate-set revalidation: a concurrent writer may have created an observation
        # referencing these exact sources during the LLM window (empty-snapshot race §6.1).
        twin = await conn.fetchval(
            f"SELECT 1 FROM {fq_table('memory_units')}"
            f" WHERE bank_id = $1 AND fact_type = 'observation' AND source_memory_ids @> $2::uuid[] LIMIT 1",
            bank_id,
            ordered,
        )
        if twin is not None:
            return "twin_exists_for_sources"
        return "ok"
    # Non-SQL store: existence + consolidated markers through the store seam.
    present = await store.get_memories(
        conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[str(mid) for mid in ordered]
    )
    live_ids = {str(m.unit_id) for m in present}
    for sid in ordered:
        sid_str = str(sid)
        if sid_str not in live_ids:
            return f"source_deleted:{sid_str}"
        m = next((x for x in present if str(x.unit_id) == sid_str), None)
        if m is not None and m.consolidated_at is not None:
            return f"source_consumed:{sid_str}"
        expected = expected_fingerprints.get(sid_str)
        if expected is not None and m is not None:
            fresh = _source_fingerprint(
                {
                    "text": m.text,
                    "fact_type": m.fact_type,
                    "tags": m.tags or [],
                    "event_date": m.event_date,
                    "occurred_start": m.occurred_start,
                    "occurred_end": m.occurred_end,
                    "mentioned_at": m.mentioned_at,
                    "proof_count": m.proof_count,
                    "source_memory_ids": m.source_memory_ids or [],
                    "consolidated_at": m.consolidated_at,
                }
            )
            if not _fingerprint_matches(expected, fresh):
                return f"source_changed:{sid_str}"
    # Non-SQL stores cannot do the SQL lineage twin check; their CAS seam + caller-owned
    # txn provides the equivalent guarantee on fold/merge paths.
    return "ok"


def _iso_or_none(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    return str(v)


def _memory_fact_from_stored(memory: StoredMemory, template: "MemoryFact | None" = None) -> "MemoryFact":
    """Rebuild a MemoryFact the LLM will see from the snapshot-backed StoredMemory."""
    from ..response_models import MemoryFact

    meta = None
    if memory.metadata is not None:
        meta = {str(k): str(v) for k, v in memory.metadata.items()}
    return MemoryFact(
        id=str(memory.unit_id),
        text=memory.text,
        fact_type=memory.fact_type,
        context=memory.context,
        occurred_start=_iso_or_none(memory.occurred_start),
        occurred_end=_iso_or_none(memory.occurred_end),
        mentioned_at=_iso_or_none(memory.mentioned_at),
        document_id=memory.document_id,
        metadata=meta,
        chunk_id=memory.chunk_id,
        tags=list(memory.tags or []) or None,
        source_fact_ids=list(memory.source_memory_ids or []) or None,
        scores=template.scores if template is not None else None,
        entities=template.entities if template is not None else None,
    )


async def _snapshot_observations(
    pool: DatabaseBackend,
    bank_id: str,
    unit_ids: list[str],
    *,
    conn=None,
) -> dict[str, MemorySnapshot]:
    """Authoritative snapshot of ``unit_ids`` keyed by unit id. Missing ids are omitted."""
    if not unit_ids:
        return {}
    store = get_memories()
    unique = list(dict.fromkeys(str(u) for u in unit_ids))

    async def _load(c) -> dict[str, MemorySnapshot]:
        snaps = await store.snapshot_memories(conn=c, fq_table=fq_table, bank_id=bank_id, unit_ids=unique)
        return {str(s.memory.unit_id): s for s in snaps}

    if conn is not None:
        return await _load(conn)
    async with acquire_with_retry(pool) as c:
        return await _load(c)


async def _bind_recalled_observations_to_snapshots(
    pool: DatabaseBackend,
    bank_id: str,
    observations: list["MemoryFact"],
) -> tuple[list["MemoryFact"], dict[str, str]]:
    """Replace recalled observations with snapshot-backed copies taken BEFORE the LLM.

    Missing rows are dropped so the LLM never sees a vanished observation. Tokens are
    those of the StoredMemory objects used to rebuild the shown facts.
    """
    if not observations:
        return [], {}
    snaps = await _snapshot_observations(pool, bank_id, [str(o.id) for o in observations])
    bound: list[MemoryFact] = []
    revisions: dict[str, str] = {}
    for obs in observations:
        snap = snaps.get(str(obs.id))
        if snap is None:
            continue
        bound.append(_memory_fact_from_stored(snap.memory, template=obs))
        revisions[str(obs.id)] = snap.revision
    return bound, revisions


async def _prevalidate_prepared_batch(
    prepared: "_PreparedBatch",
    conn: "Connection",
    bank_id: str,
) -> str:
    """Prevalidate ONE prepared plan under the Phase-B bank guard, before ANY mutation.

    Ruling 1 (batch-wide stale atomicity): validation must precede mutation so a stale
    plan aborts the ENTIRE Phase-B attempt with zero writes instead of continuing and
    salvaging the non-stale subset. This runs once per plan before the first
    delete/update/create/source-mark/witness executes.

    Validates (fresh, under the guard):
    - every observation shown to the main LLM (``observation_revisions``) still
      matches its Phase-A token — missing or mutated -> stale;
    - every CREATE's source ids — liveness, unconsumed, unchanged, no twin (the
      existing :func:`_fresh_source_validation`);
    - every UPDATE's target observation still matches ``phase_a_revision`` AND its
      source ids are live/unchanged/unconsumed;
    - every DELETE's target observation still matches ``phase_a_revision``;
    - every decision-relevant dedup candidate / fold twin still matches its Phase-A token.

    Returns ``"ok"`` when every plan is valid, or a short stale reason naming the
    first invalid element. The caller aborts the whole batch on any non-"ok".
    """

    expected_obs = dict(prepared.observation_revisions or {})
    for pupd in prepared.updates:
        if pupd.phase_a_revision:
            expected_obs.setdefault(str(pupd.update.observation_id), pupd.phase_a_revision)
    for pdel in prepared.deletes:
        if pdel.phase_a_revision:
            expected_obs.setdefault(str(pdel.delete.observation_id), pdel.phase_a_revision)
    for pcreate in prepared.creates:
        if pcreate.phase_a_target_revision and pcreate.dedup_outcome and pcreate.dedup_outcome.best_id:
            expected_obs.setdefault(str(pcreate.dedup_outcome.best_id), pcreate.phase_a_target_revision)
        for cid, crev in (pcreate.candidate_revisions or {}).items():
            expected_obs.setdefault(str(cid), crev)
        if pcreate.dedup_outcome is not None:
            for cid, crev in (pcreate.dedup_outcome.candidate_revisions or {}).items():
                expected_obs.setdefault(str(cid), crev)

    if expected_obs:
        snaps = await _snapshot_observations(pool=None, bank_id=bank_id, unit_ids=list(expected_obs), conn=conn)
        for oid, expected in expected_obs.items():
            snap = snaps.get(str(oid))
            if snap is None:
                return f"observation_missing:{oid}"
            if snap.revision != expected:
                return f"observation_stale:{oid}"

    for pcreate in prepared.creates:
        stale_reason = await _fresh_source_validation(
            conn=conn,
            bank_id=bank_id,
            source_ids=pcreate.create_source_ids or [],
            expected_fingerprints=prepared.source_snapshots,
        )
        if stale_reason != "ok":
            return f"create_stale:{stale_reason}"
        # Ruling 2: bounded semantic candidate-set revalidation (fresh in-scope twin
        # above threshold not in the Phase-A snapshot -> whole batch stale).
        semantic_reason = await _semantic_candidate_expansion(conn, bank_id, pcreate)
        if semantic_reason != "ok":
            return f"create_semantic_stale:{semantic_reason}"

    for pupd in prepared.updates:
        # Source ids must still be live/unchanged/unconsumed (target token already compared).
        upd_source_ids = [m["id"] for m in pupd.source_mems]
        stale_reason = await _fresh_source_validation(
            conn=conn,
            bank_id=bank_id,
            source_ids=upd_source_ids,
            expected_fingerprints=prepared.source_snapshots,
        )
        if stale_reason != "ok":
            return f"update_source_stale:{stale_reason}"

    return "ok"


async def _observation_exists(
    conn: "Connection",
    bank_id: str,
    observation_id: str,
) -> bool:
    """Return whether an observation row exists in the bank (fresh read under guard)."""
    store = get_memories()
    if store.writes_memory_rows_in_sql_for(bank_id):
        row = await conn.fetchval(
            f"SELECT 1 FROM {fq_table('memory_units')}"
            f" WHERE bank_id = $1 AND id = $2::uuid AND fact_type = 'observation'",
            bank_id,
            observation_id,
        )
        return row is not None
    present = await store.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[observation_id])
    return any(str(m.unit_id) == str(observation_id) for m in present)


async def _semantic_candidate_expansion(
    conn: "Connection",
    bank_id: str,
    pcreate: "_PreparedCreate",
) -> str:
    """Ruling 2: bounded semantic candidate-set revalidation under the bank guard.

    Uses the ALREADY-COMPUTED Phase-A embedding (``pcreate.embedding_str``) — no
    embedder/recall/LLM/reranker runs under the lock. Runs a bounded pgvector
    similarity lookup over in-scope observations (same exact-scope/tag predicate
    and threshold as the Phase-A dedup adjudication). If a fresh in-scope candidate
    meets the threshold but was NOT in the Phase-A candidate snapshot, a semantic
    twin was introduced during the LLM window: return a stale reason so the whole
    batch aborts (Ruling 1). Phase B only DETECTS expansion and aborts — it never
    decides a merge; the reprepare's Phase A does adjudication.

    Returns ``"ok"`` or a short stale reason.
    """
    if not pcreate.embedding_str:
        # No embedding (dedup disabled or embed failure): nothing to probe; the
        # same-source containment query in ``_fresh_source_validation`` still guards
        # the empty-snapshot race, and exact-text/lineage CAS still applies.
        return "ok"
    store = get_memories()
    if not store.writes_memory_rows_in_sql_for(bank_id):
        # Non-SQL store: no bounded pgvector lookup; its CAS seam provides the
        # equivalent guard on fold/merge paths.
        return "ok"
    threshold = get_config().consolidation_dedup_threshold
    tags = pcreate.agg.tags or []
    # Same scope predicate as ``_dedup_adjudicate``: all_strict when the CREATE has
    # tags (all must match), otherwise any observation is in scope.
    if tags:
        tags_clause = "tags @> $4::varchar[]"
        limit_param = "$5"
        tags_args: list[Any] = [tags]
    else:
        tags_clause = "TRUE"
        limit_param = "$4"
        tags_args = []
    rows = await conn.fetch(
        f"SELECT id FROM {fq_table('memory_units')}"
        f" WHERE bank_id = $1 AND fact_type = 'observation' AND embedding IS NOT NULL"
        f" AND (1 - (embedding <=> $2::vector)) >= $3"
        f" AND {tags_clause}"
        f" ORDER BY embedding <=> $2::vector LIMIT {limit_param}",
        bank_id,
        pcreate.embedding_str,
        threshold,
        *tags_args,
        _DEDUP_TOP_K,
    )
    known = set(pcreate.candidate_ids)
    fresh_hits = [str(r["id"]) for r in rows if str(r["id"]) not in known]
    if fresh_hits:
        return f"semantic_twin_new_candidate:{sorted(fresh_hits)[:5]}"
    return "ok"


class _CreateAction(BaseModel):
    text: str
    source_fact_ids: list[str]  # memory UUIDs from the NEW FACTS list
    # One-sentence justification from the LLM (why CREATE vs UPDATE). Diagnostic
    # only — surfaced in the consolidation trace to explain duplicate creates.
    reason: str = ""

    @field_validator("text", mode="before")
    @classmethod
    def sanitize_text(cls, v: str) -> str:
        return sanitize_llm_output(v) or ""

    @field_validator("source_fact_ids", mode="before")
    @classmethod
    def ensure_list(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [v]
        return v


class _UpdateAction(BaseModel):
    text: str
    observation_id: str  # UUID of the existing observation to update
    source_fact_ids: list[str]  # memory UUIDs from the NEW FACTS list
    reason: str = ""  # LLM's one-sentence justification (diagnostic only)

    @field_validator("text", mode="before")
    @classmethod
    def sanitize_text(cls, v: str) -> str:
        return sanitize_llm_output(v) or ""

    @field_validator("source_fact_ids", mode="before")
    @classmethod
    def ensure_list(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [v]
        return v


class _DeleteAction(BaseModel):
    observation_id: str  # UUID of the observation to remove
    reason: str = ""  # LLM's one-sentence justification (diagnostic only)


class _ConsolidationBatchResponse(BaseModel):
    creates: list[_CreateAction] = []
    updates: list[_UpdateAction] = []
    deletes: list[_DeleteAction] = []


@dataclass
class _BatchLLMResult:
    creates: list[_CreateAction] = field(default_factory=list)
    updates: list[_UpdateAction] = field(default_factory=list)
    deletes: list[_DeleteAction] = field(default_factory=list)
    obs_count: int = 0
    prompt_chars: int = 0
    failed: bool = False


@dataclass
class _SourceAggregation:
    """Fields inherited by an observation from its source memories."""

    event_date: datetime | None
    occurred_start: datetime | None
    occurred_end: datetime | None
    mentioned_at: datetime | None
    tags: list[str]


@dataclass
class _PreparedCreate:
    """One CREATE action prepared in Phase A, executed under CAS in Phase B.

    Carries the source ids, aggregated source fields, and the pre-computed
    embedding + dedup outcome so Phase B does no slow work under the bank guard.
    ``embedding_str`` is precomputed in Phase A (design §4.2 — no embedder under
    the bank lock).
    """

    create: _CreateAction
    source_mems: list[dict[str, Any]]
    agg: _SourceAggregation
    create_source_ids: list[uuid.UUID]
    dedup_outcome: "_DedupOutcome | None" = None
    embedding_str: str | None = None
    # Ruling 2: Phase-A candidate snapshot — observation ids the dedup adjudication
    # probed (in-scope bounded top-K). Re-checked under the bank guard so a fresh
    # semantic twin above threshold aborts the batch instead of duplicating it.
    candidate_ids: set[str] = field(default_factory=set)
    # Token of the fold twin (``dedup_outcome.best_id``) when Phase A decided to fold.
    phase_a_target_revision: str | None = None
    # Token of every probed candidate (same map as ``dedup_outcome.candidate_revisions``).
    candidate_revisions: dict[str, str] = field(default_factory=dict)


@dataclass
class _PreparedUpdate:
    """One UPDATE action prepared in Phase A, executed under CAS in Phase B."""

    update: _UpdateAction
    source_mems: list[dict[str, Any]]
    agg: _SourceAggregation
    embedding_str: str | None
    dedup_outcome: "_DedupOutcome | None" = None
    # Token of the UPDATE target as shown to the LLM (the snapshot-backed observation).
    phase_a_revision: str = ""


@dataclass
class _PreparedDelete:
    """One DELETE action prepared in Phase A, executed under CAS in Phase B."""

    delete: _DeleteAction
    # Token of the DELETE target as shown to the LLM.
    phase_a_revision: str = ""


@dataclass
class _PreparedBatch:
    """Phase-A result for one ``_process_memory_batch`` call (one observation scope).

    All slow work — recall, LLM, embeddings, dedup adjudication — completes here,
    off any write connection. Phase B (:func:`_commit_prepared_batch`) executes the
    writes under the caller-owned bank-guarded connection with CAS + fresh validation.

    ``source_snapshots`` maps each source memory id (str) to its authoritative
    Phase-A state (the mutation-relevant fields). Phase B re-locks the source rows
    under the bank guard and compares against this snapshot (design §4.2 step 3-4,
    §6.1 fresh source-ID validation) so a writer whose sources were consumed or
    mutated during the LLM window rolls back with zero writes instead of creating a
    duplicate observation.
    """

    memories: list[dict[str, Any]]
    per_fact_obs_ids: dict[str, set[str]]
    union_observations: list["MemoryFact"]
    llm_result: _BatchLLMResult
    fact_tags: list[str]
    deletes: list[_PreparedDelete]
    updates: list[_PreparedUpdate]
    creates: list[_PreparedCreate]
    dedup_enabled: bool
    dedup_llm_config: Any = None
    source_snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Token of every observation shown to the main LLM, keyed by unit id. Derived from
    # the StoredMemory snapshot taken BEFORE the LLM call (judge 76f0ac68).
    observation_revisions: dict[str, str] = field(default_factory=dict)


def _aggregate_source_fields(source_mems: list[dict[str, Any]], tags: list[str] | None = None) -> _SourceAggregation:
    """Compute the observation fields inherited from a set of source memories.

    Temporal aggregation rules:
    - ``event_date``    — earliest across sources (min)
    - ``occurred_start`` — earliest across sources (min)
    - ``occurred_end``   — latest across sources (max)
    - ``mentioned_at``   — latest across sources (max)

    Fields remain ``None`` when no source memory carries that information, so
    observations are never stamped with an artificial timestamp.

    ``tags`` defaults to those of the first source memory when not explicitly
    provided (all memories in a consolidation batch share the same tag set).
    """
    effective_tags = tags if tags is not None else (source_mems[0].get("tags") or [] if source_mems else [])
    return _SourceAggregation(
        event_date=_min_date(m.get("event_date") for m in source_mems),
        occurred_start=_min_date(m.get("occurred_start") for m in source_mems),
        occurred_end=_max_date(m.get("occurred_end") for m in source_mems),
        mentioned_at=_max_date(m.get("mentioned_at") for m in source_mems),
        tags=effective_tags,
    )


async def _count_observations_for_scope(
    conn: "Connection",
    bank_id: str,
    tags: list[str],
) -> int:
    """Count existing observations matching the given tag scope.

    Returns the count of observations whose tags contain all specified tags.
    Observations with no tags are not counted (the limit does not apply to them).
    """
    store = get_memories()
    if store.writes_memory_rows_in_sql_for(bank_id):
        return await conn.fetchval(
            f"SELECT COUNT(*) FROM {fq_table('memory_units')} "
            f"WHERE bank_id = $1 AND fact_type = 'observation' AND tags @> $2::varchar[]",
            bank_id,
            tags,
        )
    # A store that keeps observations outside Postgres: count them through it (tag containment).
    total = 0
    page_token = ""
    for _ in range(100):
        page = await store.scan_memories(
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            fact_types=["observation"],
            tags=tags or None,
            tags_match="all",
            limit=500,
            page_token=page_token,
        )
        total += len(page.memories)
        page_token = page.next_page_token
        if not page_token:
            break
    return total


@dataclass(frozen=True)
class _ScopeLimitRule:
    """One ``observation_scope_limits`` rule: a scope pattern -> an observation cap.

    ``globs`` is a tuple of fnmatch tag-globs describing one consolidation scope.
    A concrete scope (the set of ``fact_tags`` for a consolidation pass) matches
    under *exact cover*: every tag is matched by some glob AND every glob matches
    some tag. So ``["shared"]`` matches the scope ``{shared}`` but not
    ``{run_1, shared}``, and ``["run_*", "shared"]`` matches ``{run_1, shared}``
    but not ``{shared}``.

    ``limit`` is the cap applied to matching scopes (-1 = unlimited, 0 = no new
    observations, >0 = hard cap), mirroring ``max_observations_per_scope``.
    """

    globs: tuple[str, ...]
    limit: int


def _parse_scope_limit_rules(raw: Any) -> list[_ScopeLimitRule]:
    """Parse the raw ``observation_scope_limits`` config into ordered rules.

    The config round-trips as JSON through env and the bank-config API, so this
    is defensive: malformed entries are skipped rather than raising, and list
    order is preserved (first match wins in :func:`_effective_scope_limit`).
    """
    if not isinstance(raw, list):
        return []
    rules: list[_ScopeLimitRule] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        scope = entry.get("scope")
        limit = entry.get("limit")
        if not isinstance(scope, list) or not scope:
            continue
        if not all(isinstance(g, str) and g for g in scope):
            continue
        # bool is an int subclass — reject True/False masquerading as a limit.
        if not isinstance(limit, int) or isinstance(limit, bool):
            continue
        rules.append(_ScopeLimitRule(globs=tuple(scope), limit=limit))
    return rules


def _scope_matches_globs(globs: tuple[str, ...], tags: list[str]) -> bool:
    """Exact-cover match between a scope pattern and a concrete tag set.

    True iff every tag is covered by at least one glob AND every glob covers at
    least one tag (no uncovered tags, no vacuous globs). Untagged scopes never
    match, so a scope limit never applies to untagged observations (consistent
    with the ``and fact_tags`` guard at the call site). Matching is
    case-sensitive (``fnmatchcase``) for deterministic cross-platform behaviour.
    """
    tagset = set(tags)
    if not tagset:
        return False
    if not all(any(fnmatchcase(t, g) for g in globs) for t in tagset):
        return False
    if not all(any(fnmatchcase(t, g) for t in tagset) for g in globs):
        return False
    return True


def _effective_scope_limit(config: Any, fact_tags: list[str]) -> int:
    """Resolve the observation cap for one concrete consolidation scope.

    The first rule in ``observation_scope_limits`` whose pattern exact-covers
    ``fact_tags`` wins; otherwise falls back to the bank-wide
    ``max_observations_per_scope``. Wildcards live only here, matched against the
    already-resolved concrete tags — the SQL count stays exact and indexed.
    """
    if config is None:
        return -1
    for rule in _parse_scope_limit_rules(getattr(config, "observation_scope_limits", None)):
        if _scope_matches_globs(rule.globs, fact_tags):
            return rule.limit
    return config.max_observations_per_scope


def _build_response_model(
    max_creates: int | None = None,
    *,
    supports_max_items: bool = True,
) -> type[_ConsolidationBatchResponse]:
    """Build a response model, optionally constraining creates via JSON schema.

    Some structured-output backends (notably Bedrock Converse) reject the JSON
    Schema ``maxItems`` keyword emitted by Pydantic's list ``max_length``. Operators
    can disable the schema hint for those backends; the prompt capacity note and
    post-response truncation still enforce the observation cap.
    """
    if not supports_max_items or max_creates is None or max_creates < 0:
        return _ConsolidationBatchResponse

    from pydantic import Field as PydanticField

    clamped = max(max_creates, 0)

    class _ConstrainedConsolidationBatchResponse(_ConsolidationBatchResponse):
        creates: list[_CreateAction] = PydanticField(default=[], max_length=clamped)

    return _ConstrainedConsolidationBatchResponse


class ConsolidationPerfLog:
    """Performance logging for consolidation operations."""

    def __init__(self, bank_id: str):
        self.bank_id = bank_id
        self.start_time = time.time()
        self.lines: list[str] = []
        self.timings: dict[str, float] = {}
        self.timing_counts: dict[str, int] = {}
        self.llm_calls: int = 0
        self.total_obs_in_context: int = 0
        self.total_prompt_chars: int = 0

    def log(self, message: str) -> None:
        """Add a log line."""
        self.lines.append(message)

    def record_timing(self, key: str, duration: float) -> None:
        """Record a timing measurement.

        Tracks both total seconds and call count so the summary can
        distinguish one slow call from many fast calls in aggregate.
        """
        self.timings[key] = self.timings.get(key, 0.0) + duration
        self.timing_counts[key] = self.timing_counts.get(key, 0) + 1

    def record_llm_call(self, obs_count: int, prompt_chars: int) -> None:
        """Record stats for a single LLM call."""
        self.llm_calls += 1
        self.total_obs_in_context += obs_count
        self.total_prompt_chars += prompt_chars

    def merge_from(self, other: "ConsolidationPerfLog") -> None:
        """Merge a per-batch perf log into this (job-level) one.

        Used by the parallel dispatcher: each in-flight batch records into its
        own ``ConsolidationPerfLog`` so the per-batch log line shows only that
        batch's timings (no cross-batch interleaving). After the batch finishes
        we fold the local counters into the job-level perf, which then drives
        the final ``flush()`` summary.

        ``lines`` is intentionally NOT merged — log lines are emitted directly
        in ``logger.info`` calls by the dispatcher; the perf object's ``lines``
        buffer is only used by the top-level job summary.
        """
        for key, value in other.timings.items():
            self.timings[key] = self.timings.get(key, 0.0) + value
        for key, count in other.timing_counts.items():
            self.timing_counts[key] = self.timing_counts.get(key, 0) + count
        self.llm_calls += other.llm_calls
        self.total_obs_in_context += other.total_obs_in_context
        self.total_prompt_chars += other.total_prompt_chars

    def flush(self) -> None:
        """Flush all log lines to the logger."""
        total_time = time.time() - self.start_time
        header = f"\n{'=' * 60}\nCONSOLIDATION for bank {self.bank_id}"
        footer = f"{'=' * 60}\nCONSOLIDATION COMPLETE: {total_time:.3f}s total\n{'=' * 60}"

        log_output = header + "\n" + "\n".join(self.lines) + "\n" + footer
        logger.info(log_output)


def _as_dt(v: "datetime | str | None") -> "datetime | None":
    """Coerce an ISO string to a datetime. Recall results can carry timestamps as strings while
    the store's addressed reads hand back datetimes, so normalise before comparing."""
    return datetime.fromisoformat(v) if isinstance(v, str) else v


def _merge_min(a: "datetime | str | None", b: "datetime | str | None") -> "datetime | None":
    """SQL ``LEAST(a, COALESCE(b, a))`` in Python: the earlier of two times, ignoring None."""
    a, b = _as_dt(a), _as_dt(b)
    return a if b is None else b if a is None else min(a, b)


def _merge_max(a: "datetime | str | None", b: "datetime | str | None") -> "datetime | None":
    """SQL ``GREATEST(a, COALESCE(b, a))`` in Python: the later of two times, ignoring None."""
    a, b = _as_dt(a), _as_dt(b)
    return a if b is None else b if a is None else max(a, b)


async def _reconcile_merge_via_store(
    store,
    conn,
    memory_engine: "MemoryEngine",
    bank_id: str,
    observation_id: str,
    merged_text: str,
    add_source_ids: list,
    txn=None,
) -> None:
    """Dedup merge for a store that owns its rows: fold the extra source facts and the merged text
    into the twin observation and re-upsert it, preserving its other fields. Re-embeds the merged
    text because ``get_memories`` does not return the stored vector (the SQL path reuses it in
    place instead)."""
    current = await store.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[observation_id])
    cur = current[0] if current else None
    if cur is None:
        return
    merged_sources = list(dict.fromkeys([*(cur.source_memory_ids or []), *(str(s) for s in add_source_ids)]))
    embeddings = await embedding_utils.generate_embeddings_batch(memory_engine.embeddings, [merged_text])
    await store.upsert_observation(
        conn=conn,
        bank_id=bank_id,
        txn=txn,
        record=FactRecord(
            unit_id=observation_id,
            text=merged_text,
            embedding=str(embeddings[0]) if embeddings else None,
            fact_type="observation",
            tags=list(cur.tags or []),
            proof_count=len(merged_sources),
            source_memory_ids=merged_sources,
            event_date=cur.event_date,
            occurred_start=cur.occurred_start,
            occurred_end=cur.occurred_end,
            mentioned_at=cur.mentioned_at,
            created_at=cur.created_at,
        ),
    )


async def _fetch_unconsolidated_rows(
    conn,
    bank_id: str,
    fact_types: list[str],
    limit: int,
    observation_scopes: list[list[str]] | None,
) -> list[dict[str, Any]]:
    """Unconsolidated candidate facts, read through the memories store.

    The store owns the memories, so this must ask it rather than query ``memory_units``
    directly — otherwise a store that keeps its rows elsewhere yields nothing and
    consolidation silently produces no observations. Returns the same row-dict shape the
    consolidation loop consumes. Mirrors the job's scope filter: with scopes, OR each
    "tags ⊇ scope" and merge oldest-first; without, one unscoped read.
    """
    store = get_memories()
    scopes: list[list[str] | None] = list(observation_scopes) if observation_scopes else [None]
    by_id: dict[str, Any] = {}
    for scope in scopes:
        for m in await store.find_unconsolidated(
            conn=conn, fq_table=fq_table, bank_id=bank_id, fact_types=fact_types, limit=limit, scope_tags=scope
        ):
            by_id.setdefault(m.unit_id, m)
    ordered = sorted(by_id.values(), key=lambda m: (m.created_at is None, m.created_at))[:limit]
    return [
        {
            "id": uuid.UUID(m.unit_id),
            "text": m.text,
            "fact_type": m.fact_type,
            "occurred_start": m.occurred_start,
            "occurred_end": m.occurred_end,
            "event_date": m.event_date,
            "tags": list(m.tags or []),
            "mentioned_at": m.mentioned_at,
            "observation_scopes": m.observation_scopes,
        }
        for m in ordered
    ]


async def _refetch_source_rows(
    conn,
    bank_id: str,
    unit_ids: list[uuid.UUID],
) -> list[dict[str, Any]]:
    """Ruling 3: re-fetch original source rows FRESH by id for a bounded reprepare.

    Returns the same row-dict shape :func:`_fetch_unconsolidated_rows` produces, but
    reads by explicit unit ids (including already-consolidated rows, so the reprepare's
    Phase A sees a consumed source as such rather than the stale pre-abort snapshot).
    Missing ids are dropped (a deleted source can no longer be reprepared).
    """
    store = get_memories()
    if not store.writes_memory_rows_in_sql_for(bank_id):
        present = await store.get_memories(
            conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[str(uid) for uid in unit_ids]
        )
        by_id: dict[str, StoredMemory] = {str(m.unit_id): m for m in present}
    else:
        rows = await conn.fetch(
            f"SELECT id, text, fact_type, event_date, occurred_start, occurred_end, mentioned_at,"
            f" tags, observation_scopes FROM {fq_table('memory_units')}"
            f" WHERE bank_id = $1 AND id = ANY($2::uuid[])",
            bank_id,
            list(unit_ids),
        )
        by_id = {
            str(r["id"]): StoredMemory(
                unit_id=str(r["id"]),
                text=r["text"],
                fact_type=r["fact_type"],
                event_date=r["event_date"],
                occurred_start=r["occurred_start"],
                occurred_end=r["occurred_end"],
                mentioned_at=r["mentioned_at"],
                tags=list(r["tags"] or []),
                observation_scopes=r["observation_scopes"],
            )
            for r in rows
        }
    return [
        {
            "id": uuid.UUID(str(m.unit_id)),
            "text": m.text,
            "fact_type": m.fact_type,
            "occurred_start": m.occurred_start,
            "occurred_end": m.occurred_end,
            "event_date": m.event_date,
            "tags": list(m.tags or []),
            "mentioned_at": m.mentioned_at,
            "observation_scopes": m.observation_scopes,
        }
        for uid in unit_ids
        if (m := by_id.get(str(uid))) is not None
    ]


#: Cap on the store-side count of unconsolidated facts. Used only for the "is there work?"
#: gate and progress reporting, so a floor at this size is harmless on a huge backlog.
_COUNT_LIMIT = 100_000


async def _count_unconsolidated_rows(
    conn,
    bank_id: str,
    fact_types: list[str],
    observation_scopes: list[list[str]] | None,
) -> int:
    """Count of unconsolidated candidate facts, from the store (bounded by ``_COUNT_LIMIT``).

    Asks the store for a *count* rather than fetching the rows and taking ``len`` — on the SQL
    store that is one bounded ``COUNT(*)`` instead of shipping up to ``_COUNT_LIMIT`` full memory
    rows across the wire on every job start / progress tick.
    """
    scopes: list[list[str] | None] = list(observation_scopes) if observation_scopes else [None]
    return await get_memories().count_unconsolidated(
        conn=conn, fq_table=fq_table, bank_id=bank_id, fact_types=fact_types, scopes=scopes, limit=_COUNT_LIMIT
    )


def _as_op_uuid(operation_id: str | uuid.UUID) -> uuid.UUID:
    return uuid.UUID(operation_id) if isinstance(operation_id, str) else operation_id


async def _persist_pending_refresh_tags(conn, operation_id: str, new_tags: list[str]) -> None:
    """Union ``new_tags`` into the consolidation op's durable ``pending_refresh_tags``.

    Called inside each batch's witness transaction, so the tags of an
    already-consolidated batch are durable the instant that batch is — a mid-round
    worker crash no longer loses them. On retry the op re-reads ``task_payload`` and the
    final round still refreshes those models (#3411); without this, a crash after batch 1
    committed but before the round finished would drop batch 1's tags, because the retry
    skips its now-consolidated rows and never re-collects them. ``SELECT ... FOR UPDATE``
    serialises the concurrent batches of one op so their unions don't clobber each other.
    """
    op_uuid = _as_op_uuid(operation_id)
    row = await conn.fetchrow(
        f"SELECT task_payload FROM {fq_table('async_operations')} WHERE operation_id = $1 FOR UPDATE",
        op_uuid,
    )
    if row is None:
        return
    payload = row["task_payload"]
    payload = json.loads(payload) if isinstance(payload, str) else (payload or {})
    existing = set(payload.get("pending_refresh_tags") or [])
    merged = existing | set(new_tags)
    if merged == existing:
        return
    payload["pending_refresh_tags"] = sorted(merged)
    await conn.execute(
        f"UPDATE {fq_table('async_operations')} SET task_payload = $1::jsonb, updated_at = now() "
        f"WHERE operation_id = $2",
        json.dumps(payload),
        op_uuid,
    )


async def _read_pending_refresh_tags(pool, operation_id: str) -> set[str]:
    """Read the op's durably-accumulated ``pending_refresh_tags`` (crash-safe source of
    truth for the final-round flush)."""
    async with acquire_with_retry(pool) as conn:
        row = await conn.fetchrow(
            f"SELECT task_payload FROM {fq_table('async_operations')} WHERE operation_id = $1",
            _as_op_uuid(operation_id),
        )
    if row is None:
        return set()
    payload = row["task_payload"]
    payload = json.loads(payload) if isinstance(payload, str) else (payload or {})
    return set(payload.get("pending_refresh_tags") or [])


async def run_consolidation_job(
    memory_engine: "MemoryEngine",
    bank_id: str,
    request_context: "RequestContext",
    operation_id: str | None = None,
    observation_scopes: list[list[str]] | None = None,
    pending_refresh_tags: list[str] | None = None,
) -> dict[str, Any]:
    """Run consolidation job for a bank.

    Store-CAS consolidation v1 is implemented and proven only on PostgreSQL. Oracle is
    rejected before bank config resolution, LLM setup, backend acquisition, or Phase A/B;
    silently skipping would leave unconsolidated memories while appearing successful.

    This is called after retain operations to consolidate new memories into mental models.

    Args:
        memory_engine: MemoryEngine instance
        bank_id: Bank identifier
        request_context: Request context for authentication
        operation_id: Optional operation ID for tracking
        observation_scopes: Optional list of tag scopes. When provided, only
            unconsolidated memories whose tags contain all tags in at least one
            scope are processed.
        pending_refresh_tags: Tags of memories consolidated by earlier rounds of this
            round-limited chain, carried through the re-queue so the final round can
            refresh every affected mental model exactly once (#3411).

    Returns:
        Dict with consolidation results
    """
    database_backend = get_config().database_backend
    if database_backend == "oracle":
        raise UnsupportedConsolidationDialectError(
            "Store-CAS consolidation v1 is PostgreSQL-only; Oracle consolidation is not supported"
        )

    # Resolve bank-specific config with hierarchical overrides
    config = await memory_engine._config_resolver.resolve_full_config(bank_id, request_context)

    # Build a configured LLM wrapper that applies per-bank settings (e.g. safety settings)
    # to every call without leaking across operations.
    llm_config = memory_engine._consolidation_llm_config.with_config(config, bank_id=bank_id, operation="consolidation")

    # Bind the operation trace context for the whole run so the create/update DB
    # sites (deep inside _process_memory_batch) can accumulate the observations
    # this consolidation produced and the source memories it consumed onto the
    # trace — flushed onto every trace row on exit by attach_memory_ids.
    trace_ctx = trace_context_of(llm_config)
    trace_token = set_trace_context(trace_ctx) if trace_ctx is not None else None
    try:
        return await _run_consolidation_job(
            memory_engine,
            bank_id,
            request_context,
            config,
            llm_config,
            operation_id,
            observation_scopes,
            pending_refresh_tags,
        )
    finally:
        if trace_token is not None:
            reset_trace_context(trace_token)
            # Fire-and-forget: patched on a background task, off the consolidation
            # critical path.
            memory_engine._llm_recorder.attach_memory_ids(trace_ctx)


async def _run_consolidation_job(
    memory_engine: "MemoryEngine",
    bank_id: str,
    request_context: "RequestContext",
    config: Any,
    llm_config: Any,
    operation_id: str | None = None,
    observation_scopes: list[list[str]] | None = None,
    pending_refresh_tags: list[str] | None = None,
) -> dict[str, Any]:
    """Core consolidation flow. See ``run_consolidation_job`` for the public entrypoint."""
    perf = ConsolidationPerfLog(bank_id)
    max_memories_per_batch = config.consolidation_batch_size
    max_memories_per_round = config.consolidation_max_memories_per_round
    llm_batch_size = max(1, config.consolidation_llm_batch_size)

    # Check if consolidation is enabled
    if not config.enable_observations:
        logger.debug(f"Consolidation disabled for bank {bank_id}")
        return {"status": "disabled", "bank_id": bank_id}

    pool = memory_engine._backend

    # Get bank profile
    async with acquire_with_retry(pool) as conn:
        t0 = time.time()
        bank_row = await conn.fetchrow(
            f"""
            SELECT bank_id, name
            FROM {fq_table("banks")}
            WHERE bank_id = $1
            """,
            bank_id,
        )

        if not bank_row:
            logger.warning(f"Bank {bank_id} not found for consolidation")
            return {"status": "bank_not_found", "bank_id": bank_id}

        perf.record_timing("fetch_bank", time.time() - t0)

        # Count total unconsolidated memories for progress logging — through the store.
        total_count = await _count_unconsolidated_rows(conn, bank_id, ["experience", "world"], observation_scopes)

    if total_count == 0:
        logger.debug(f"No new memories to consolidate for bank {bank_id}")
        return {"status": "no_new_memories", "bank_id": bank_id, "memories_processed": 0}

    logger.info(f"[CONSOLIDATION] bank={bank_id} total_unconsolidated={total_count}")
    perf.log(f"[1] Found {total_count} pending memories to consolidate")

    # Initial durable progress snapshot so an operator polling the operation status
    # API sees the job has started and how much work it found, before the first batch
    # of LLM work completes (which can take minutes on a dense bank). Uses the same
    # "consolidating" stage as the per-batch heartbeat so the operator sees a single
    # phase advancing 0/N -> N/N rather than an opaque "scanning" -> "processing" hop.
    set_stage("consolidation.consolidating")
    await memory_engine._write_operation_progress(operation_id, stage="consolidating", processed=0, total=total_count)

    async def _count_unconsolidated() -> int:
        """Re-count memories still pending consolidation in this job's scope.

        ``total_count`` is a point-in-time estimate from job start; memories retained
        while consolidation runs get picked up by later fetches, so processed can pass
        it. When that happens we re-count to report a real total (processed + remaining)
        instead of pinning the bar at 100%."""
        async with acquire_with_retry(pool) as count_conn:
            return await _count_unconsolidated_rows(count_conn, bank_id, ["experience", "world"], observation_scopes)

    async def _progress_total(processed: int) -> int:
        # Cheap path: while we're still within the start-of-job estimate it's exact, so
        # no extra query. Only re-count once the estimate is exhausted (≈the final batch
        # normally, or repeatedly only if memories keep arriving mid-run).
        if processed < total_count:
            return total_count
        return processed + await _count_unconsolidated()

    # Process each memory with individual commits for crash recovery
    stats: dict[str, int] = {
        "memories_processed": 0,
        "observations_created": 0,
        "observations_updated": 0,
        "observations_merged": 0,
        "observations_deleted": 0,
        "actions_executed": 0,
        "skipped": 0,
        "memories_failed": 0,
    }

    # Track all unique tags from consolidated memories for mental model refresh filtering
    consolidated_tags: set[str] = set()

    round_limit_enabled = max_memories_per_round > 0
    round_remaining = max_memories_per_round if round_limit_enabled else float("inf")
    hit_round_limit = False

    llm_batch_num = 0
    # Cumulative counters across the whole job, shared by the per-batch log and the
    # durable progress snapshot so both report processed/total (and observation
    # tallies) under parallelism. Mutable container so the inner closure can update
    # without a `nonlocal`.
    cumulative_progress = {
        "processed": 0,
        "observations_created": 0,
        "observations_updated": 0,
        "observations_merged": 0,
        "observations_deleted": 0,
        "memories_failed": 0,
    }
    # Ruling 3: memory ids whose batch exhausted its stale-reprepare budget this job.
    # Excluded from every subsequent fetch in this invocation (no busy loop); they stay
    # unconsolidated+unfailed for a LATER job invocation.
    retry_exhausted: set[str] = set()
    while True:
        # Cap fetch size by remaining round budget
        fetch_limit = (
            min(max_memories_per_batch, int(round_remaining)) if round_limit_enabled else max_memories_per_batch
        )

        # Fetch next batch of unconsolidated memories — through the store, so a store that
        # keeps its rows outside Postgres is read too.
        async with acquire_with_retry(pool) as conn:
            t0 = time.time()
            memories = await _fetch_unconsolidated_rows(
                conn, bank_id, ["experience", "world"], fetch_limit, observation_scopes
            )
            perf.record_timing("fetch_memories", time.time() - t0)

        if not memories:
            break  # No more unconsolidated memories

        # Ruling 3 no-busy-loop: drop any batch whose stale-reprepare budget already
        # exhausted this job so we never immediately re-fetch it in the same invocation.
        memories = [m for m in memories if str(m["id"]) not in retry_exhausted]
        if not memories:
            break  # Nothing new left to try this invocation.

        # Group memories by exact tag set before batching — security requirement:
        # memories with different tags must never share an LLM call.
        tag_groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for m in memories:
            tag_key = tuple(sorted(m.get("tags") or []))
            tag_groups.setdefault(tag_key, []).append(dict(m))

        # Split each tag group into LLM batches respecting llm_batch_size, keeping
        # the group boundary intact so the dispatcher can parallelise across
        # distinct groups while running each group's batches serially.
        grouped_batches: list[list[list[dict[str, Any]]]] = []
        for group in tag_groups.values():
            grouped_batches.append([group[i : i + llm_batch_size] for i in range(0, len(group), llm_batch_size)])

        # Compute each group's union write-scope set. Used below to acquire
        # per-scope locks: any two groups whose write-scope sets share a scope S
        # will serialise on the lock for S, leaving truly disjoint groups to run
        # concurrently. We union over every memory because per-memory
        # observation_scopes can differ within a group.
        group_scopes: list[list[frozenset[str]]] = []
        for batches in grouped_batches:
            scopes: set[frozenset[str]] = set()
            for batch in batches:
                for memory in batch:
                    scopes.update(_resolve_write_scopes(memory))
            group_scopes.append(sorted(scopes, key=_scope_sort_key))

        async def _process_one_llm_batch(llm_batch_local: list[dict[str, Any]], batch_num_local: int) -> _BatchDeltas:
            """Process one LLM batch independently under the Phase A / Phase B split.

            Each batch records timings/llm-call counters into its OWN
            ``ConsolidationPerfLog`` so the per-batch log line reflects only
            this batch's work — not interleaved timings from concurrent batches
            sharing the global ``perf``. The local perf is merged into the
            job-level ``perf`` once at the end so the final summary still totals
            everything.

            Orchestration (design §4.2):
            * **Phase A — prepare**: every sub-batch × every observation scope runs
              ``_prepare_memory_batch`` (recall + LLM + embeddings + dedup adjudication)
              OFF any write connection, with no bank lock or transaction held. Adaptive
              split on LLM failure happens here, exactly as before.
            * **Phase B — commit**: under a single bank-row ``FOR UPDATE`` guard and one
              caller-owned write-group transaction, all prepared plans are committed via
              ``_commit_prepared_batch`` (fresh validation + CAS), then source marks,
              the witness row, and ``decide_txn`` all share that one transaction's fate.
            * **Ruling 3 — bounded reprepare**: if Phase B aborts stale, the whole batch
              re-fetches its sources FRESH and re-runs complete Phase A outside the lock,
              then re-enters a fresh Phase-B transaction+guard. At most
              ``config.consolidation_reprepare_attempts`` reprepares (total Phase-A runs =
              1 + that). Exhausted batches stop in this job, leave sources
              unconsolidated+unfailed, and are NOT counted processed/failed/skipped —
              surfaced via ``retry_exhausted_ids`` so the outer loop never re-fetches them
              in the same invocation (no busy loop).
            """
            llm_batch_start = time.time()
            batch_perf = ConsolidationPerfLog(bank_id)

            local_tags: set[str] = set()
            for memory in llm_batch_local:
                memory_tags = memory.get("tags") or []
                if memory_tags:
                    local_tags.update(memory_tags)

            _txn_provider = get_memories()
            # One cross-store write-group per ATTEMPT. A stale abort publishes nothing; a
            # reprepare mints a fresh handle for its own writes so an unpublished group is
            # never carried forward.
            _batch_txn = await _txn_provider.mint_txn(bank_id=bank_id, mutating=True)

            reprepare_budget = int(getattr(config, "consolidation_reprepare_attempts", 1) or 0)
            max_attempts = 1 + reprepare_budget

            async def _prepare_once(
                source_batch: list[dict[str, Any]],
            ) -> tuple[list["_PreparedBatch"], list[Any], list[Any], list[dict[str, Any]]]:
                """Phase A for one attempt (off-connection): returns plans + source intent."""
                plans: list["_PreparedBatch"] = []
                succ: list[Any] = []
                fail: list[Any] = []
                results: list[dict[str, Any]] = []
                pending: list[list[dict[str, Any]]] = [source_batch]
                while pending:
                    sub_batch = pending.pop(0)
                    obs_tags_list = _resolve_obs_tags_list(sub_batch[0]) if sub_batch else None

                    sub_llm_failed = False
                    sub_prepared: list["_PreparedBatch"] = []
                    if obs_tags_list:
                        for obs_tags in obs_tags_list:
                            prepared = await _prepare_memory_batch(
                                pool=pool,
                                memory_engine=memory_engine,
                                llm_config=llm_config,
                                bank_id=bank_id,
                                memories=sub_batch,
                                request_context=request_context,
                                perf=batch_perf,
                                config=config,
                                obs_tags_override=obs_tags,
                            )
                            sub_prepared.append(prepared)
                            sub_llm_failed = sub_llm_failed or prepared.llm_result.failed
                    else:
                        prepared = await _prepare_memory_batch(
                            pool=pool,
                            memory_engine=memory_engine,
                            llm_config=llm_config,
                            bank_id=bank_id,
                            memories=sub_batch,
                            request_context=request_context,
                            perf=batch_perf,
                            config=config,
                        )
                        sub_prepared.append(prepared)
                        sub_llm_failed = prepared.llm_result.failed

                    if sub_llm_failed and len(sub_batch) > 1:
                        mid = len(sub_batch) // 2
                        logger.warning(
                            f"[CONSOLIDATION] bank={bank_id} LLM failed for sub-batch of {len(sub_batch)},"
                            f" splitting into {mid}/{len(sub_batch) - mid}"
                        )
                        pending[0:0] = [sub_batch[:mid], sub_batch[mid:]]
                    elif sub_llm_failed:
                        fail.append(sub_batch[0]["id"])
                        results.append({"action": "failed"})
                        logger.warning(
                            f"[CONSOLIDATION] bank={bank_id} LLM failed for single memory"
                            f" {sub_batch[0]['id']}, marking consolidation_failed_at"
                        )
                    else:
                        plans.extend(sub_prepared)
                        # A successfully-prepared sub-batch contributes every memory to the
                        # succeeded set (matching the original: only LLM failure keeps a memory
                        # unmarked / failed; skipped-but-prepared memories are still consolidated).
                        succ.extend(m["id"] for m in sub_batch)
                return plans, succ, fail, results

            all_results: list[dict[str, Any]] = []
            all_deleted = 0
            succeeded_ids: list[Any] = []
            failed_ids: list[Any] = []
            stale_ids: set[str] = set()
            retry_exhausted_ids: set[str] = set()
            batch_stale_reason: str | None = None
            current_source_batch: list[dict[str, Any]] = llm_batch_local

            try:
                for attempt in range(1, max_attempts + 1):
                    if attempt > 1:
                        # Ruling 3 reprepare: re-fetch the ORIGINAL source rows FRESH (outside any
                        # lock/txn), then re-run complete Phase A off-connection.
                        async with acquire_with_retry(pool) as rconn:
                            current_source_batch = await _refetch_source_rows(
                                rconn,
                                bank_id,
                                [uuid.UUID(str(m["id"])) for m in llm_batch_local],
                            )
                        if not current_source_batch:
                            logger.info(
                                f"[CONSOLIDATION] bank={bank_id} llm_batch #{batch_num_local} reprepare:"
                                f" all sources gone; leaving retry-exhausted"
                            )
                            retry_exhausted_ids.update(str(m["id"]) for m in llm_batch_local)
                            break
                        # Fresh write-group for this attempt's writes.
                        try:
                            await _txn_provider.decide_txn(_batch_txn, commit=False)
                        except Exception:
                            logger.warning(
                                f"[CONSOLIDATION] bank={bank_id} failed to abort stale write-group"
                                f" for llm_batch #{batch_num_local}; recovery sweep will resolve it",
                                exc_info=True,
                            )
                        _batch_txn = await _txn_provider.mint_txn(bank_id=bank_id, mutating=True)

                    # Reset per-attempt accumulators so a stale abort contributes nothing.
                    all_results = []
                    all_deleted = 0
                    succeeded_ids = []
                    failed_ids = []
                    stale_ids = set()
                    batch_stale_reason = None

                    # ---- Phase A: prepare all sub-batches × scopes (slow work, off-connection) ----
                    prepared_plans, succeeded_ids, failed_ids, all_results = await _prepare_once(current_source_batch)

                    # ---- Phase B: commit all prepared plans under ONE bank guard + txn ----
                    # When every sub-batch failed at the LLM (no plans) we still take the guard so
                    # the failed marks share one logical write-group fate with the witness.
                    store = get_memories()
                    now = datetime.now(timezone.utc)
                    async with acquire_with_retry(pool) as conn:
                        # The single bank-level commit guard (design §4.1): one FOR UPDATE row lock.
                        # No lease table, no advisory lock — released by commit/rollback/teardown.
                        try:
                            async with conn.transaction():
                                await conn.execute(
                                    f"SELECT bank_id FROM {fq_table('banks')} WHERE bank_id = $1 FOR UPDATE",
                                    bank_id,
                                )
                                # Ruling 1: validation MUST precede mutation. Prevalidate every
                                # prepared plan under the guard BEFORE any delete/update/create/
                                # source-mark/witness executes. If any plan is stale, RAISE so the
                                # ``async with conn.transaction()`` context manager rolls back the
                                # ENTIRE Phase-B attempt — every write made by earlier plans included.
                                # No marks/witness/refresh-tags/decide. We do NOT salvage the non-stale
                                # subset into a partial commit.
                                if prepared_plans:
                                    for prepared in prepared_plans:
                                        reason = await _prevalidate_prepared_batch(
                                            prepared=prepared, conn=conn, bank_id=bank_id
                                        )
                                        if reason != "ok":
                                            logger.warning(
                                                f"[CONSOLIDATION] bank={bank_id} Phase-B prevalidation stale ({reason}); aborting whole batch with zero writes"
                                            )
                                            raise _BatchStaleError(reason)
                                for prepared in prepared_plans:
                                    presults, pdeleted, pstale = await _commit_prepared_batch(
                                        prepared=prepared,
                                        pool=pool,
                                        memory_engine=memory_engine,
                                        bank_id=bank_id,
                                        config=config,
                                        perf=batch_perf,
                                        txn=_batch_txn,
                                        conn=conn,
                                    )
                                    if pstale:
                                        # CAS stale during mutation (final safety net): earlier plans
                                        # may already have written rows in THIS transaction; raising
                                        # rolls them all back together (one-batch/one-fate).
                                        logger.warning(
                                            f"[CONSOLIDATION] bank={bank_id} CAS stale during mutation ({sorted(pstale)[:5]}); aborting whole batch"
                                        )
                                        raise _BatchStaleError(f"cas_stale:{sorted(pstale)[:5]}")
                                    all_deleted += pdeleted
                                    stale_ids |= pstale
                                    if not all_results:
                                        all_results.extend(presults)
                                    else:
                                        if len(presults) != len(all_results):
                                            all_results.extend(presults)
                                        else:
                                            for i, (existing, new) in enumerate(zip(all_results, presults)):
                                                if (
                                                    existing.get("action") == "skipped"
                                                    and new.get("action") != "skipped"
                                                ):
                                                    all_results[i] = new
                                                elif (
                                                    existing.get("action") != "skipped"
                                                    and new.get("action") != "skipped"
                                                ):
                                                    existing_created = existing.get(
                                                        "created", 1 if existing.get("action") == "created" else 0
                                                    )
                                                    existing_updated = existing.get(
                                                        "updated", 1 if existing.get("action") == "updated" else 0
                                                    )
                                                    new_created = new.get(
                                                        "created", 1 if new.get("action") == "created" else 0
                                                    )
                                                    new_updated = new.get(
                                                        "updated", 1 if new.get("action") == "updated" else 0
                                                    )
                                                    total = (
                                                        existing_created + existing_updated + new_created + new_updated
                                                    )
                                                    all_results[i] = {
                                                        "action": "multiple",
                                                        "created": existing_created + new_created,
                                                        "updated": existing_updated + new_updated,
                                                        "merged": 0,
                                                        "total_actions": total,
                                                    }
                                # Only a fully-validated batch may be marked / witnessed / decided.
                                # Marks+witness also run when there are no prepared plans but LLM
                                # failures occurred (failed_ids) — the all-LLM-failed case still needs
                                # its failed marks + witness to share one logical write-group fate.
                                if prepared_plans or failed_ids:
                                    effective_succeeded = [
                                        mem_id for mem_id in succeeded_ids if str(mem_id) not in stale_ids
                                    ]
                                    if effective_succeeded:
                                        await store.mark_consolidated(
                                            conn=conn,
                                            fq_table=fq_table,
                                            bank_id=bank_id,
                                            unit_ids=[str(mem_id) for mem_id in effective_succeeded],
                                            when=now,
                                            failed=False,
                                            txn=_batch_txn,
                                        )
                                    if failed_ids:
                                        await store.mark_consolidated(
                                            conn=conn,
                                            fq_table=fq_table,
                                            bank_id=bank_id,
                                            unit_ids=[str(mem_id) for mem_id in failed_ids],
                                            when=now,
                                            failed=True,
                                            txn=_batch_txn,
                                        )
                                    await _txn_provider.write_txn_witness(_batch_txn, conn=conn, fq_table=fq_table)
                                    # Persist this batch's mental-model refresh tags atomically with
                                    # the witness (#3411). Only succeeded sources contribute a tag.
                                    if operation_id and effective_succeeded:
                                        succeeded_set = {str(mem_id) for mem_id in effective_succeeded}
                                        batch_tags = sorted(
                                            {
                                                t
                                                for m in llm_batch_local
                                                if str(m["id"]) in succeeded_set
                                                for t in (m.get("tags") or [])
                                            }
                                        )
                                        if batch_tags:
                                            await _persist_pending_refresh_tags(conn, operation_id, batch_tags)
                        except _BatchStaleError as e:
                            # The ``async with conn.transaction()`` rolled back every write made so
                            # far (prevalidation-abort or mutation-CAS-abort). Discard buffered
                            # results/deleted counters so the stale attempt contributes nothing.
                            batch_stale_reason = e.reason
                            all_results.clear()
                            all_deleted = 0

                    if batch_stale_reason is not None:
                        # Ruling 1 + Ruling 3: a stale batch rolled back with zero writes and no marks/
                        # witness. Abort/unpublish the write-group (no committed fate) and give NO
                        # progress credit — sources stay unconsolidated+unfailed and remain eligible for
                        # a bounded reprepare OUTSIDE the lock. If reprepare budget remains, loop back;
                        # otherwise exhaust and stop this batch in this job.
                        try:
                            await _txn_provider.decide_txn(_batch_txn, commit=False)
                        except Exception:
                            logger.warning(
                                f"[CONSOLIDATION] bank={bank_id} failed to abort write-group for stale"
                                f" llm_batch #{batch_num_local}; recovery sweep will resolve it",
                                exc_info=True,
                            )
                        logger.info(
                            f"[CONSOLIDATION] bank={bank_id} llm_batch #{batch_num_local} aborted"
                            f" stale (zero writes, no progress credit); reason={batch_stale_reason}"
                        )
                        if attempt < max_attempts:
                            continue  # bounded reprepare outside the lock (Ruling 3)
                        retry_exhausted_ids.update(str(m["id"]) for m in llm_batch_local)
                        break

                    break  # success — no further attempts

                if retry_exhausted_ids:
                    # Ruling 3: this batch exhausted its stale-reprepare budget (or all its
                    # sources vanished). Nothing committed on any attempt — every write was
                    # rolled back stale or never attempted. Return zero-progress deltas plus
                    # the exhausted ids so the outer loop never re-fetches them this job.
                    # ``succeeded_ids``/``failed_ids`` hold Phase-A INTENT that is
                    # re-populated even on stale attempts, so they must not gate this return.
                    return _BatchDeltas(
                        stats={
                            "memories_processed": 0,
                            "observations_created": 0,
                            "observations_updated": 0,
                            "observations_merged": 0,
                            "observations_deleted": 0,
                            "actions_executed": 0,
                            "skipped": 0,
                            "memories_failed": 0,
                        },
                        tags=local_tags,
                        cancelled=False,
                        retry_exhausted_ids=retry_exhausted_ids,
                    )

                cancelled_local = False

                # ---- Post-commit bookkeeping (no writes on the guarded conn) ----
                # Note: when prepared_plans was empty we still decide/abort the txn below.
                if operation_id and not await memory_engine._check_op_alive(operation_id):
                    logger.info(
                        f"[CONSOLIDATION] bank={bank_id} operation {operation_id} cancelled (bank deleted), stopping early"
                    )
                    cancelled_local = True

                # ---- Per-batch local stats ----
                local_stats: dict[str, int] = {
                    "memories_processed": 0,
                    "observations_created": 0,
                    "observations_updated": 0,
                    "observations_merged": 0,
                    "observations_deleted": all_deleted,
                    "actions_executed": 0,
                    "skipped": 0,
                    "memories_failed": 0,
                }
                for result in all_results:
                    local_stats["memories_processed"] += 1
                    action = result.get("action")
                    if action == "created":
                        local_stats["observations_created"] += 1
                        local_stats["actions_executed"] += 1
                    elif action == "updated":
                        local_stats["observations_updated"] += 1
                        local_stats["actions_executed"] += 1
                    elif action == "merged":
                        local_stats["observations_merged"] += 1
                        local_stats["actions_executed"] += 1
                    elif action == "multiple":
                        local_stats["observations_created"] += result.get("created", 0)
                        local_stats["observations_updated"] += result.get("updated", 0)
                        local_stats["observations_merged"] += result.get("merged", 0)
                        local_stats["actions_executed"] += result.get("total_actions", 0)
                    elif action == "skipped":
                        local_stats["skipped"] += 1
                    elif action == "failed":
                        local_stats["memories_failed"] += 1

                cumulative_progress["processed"] += local_stats["memories_processed"]
                cumulative_progress["observations_created"] += local_stats["observations_created"]
                cumulative_progress["observations_updated"] += local_stats["observations_updated"]
                cumulative_progress["observations_merged"] += local_stats["observations_merged"]
                cumulative_progress["observations_deleted"] += local_stats["observations_deleted"]
                cumulative_progress["memories_failed"] += local_stats["memories_failed"]
                cum_processed = cumulative_progress["processed"]
                cum_snapshot = dict(cumulative_progress)

                llm_batch_time = time.time() - llm_batch_start
                timing_parts = [
                    f"{key}={batch_perf.timings[key]:.3f}s"
                    for key in ("recall", "llm", "embedding", "db_write")
                    if key in batch_perf.timings
                ]
                input_tokens = int(batch_perf.total_prompt_chars / 4)
                logger.info(
                    f"[CONSOLIDATION] bank={bank_id} llm_batch #{batch_num_local}"
                    f" ({len(llm_batch_local)} memories, {batch_perf.llm_calls} llm calls)"
                    f" | processed={cum_processed}/{total_count}"
                    f" | {', '.join(timing_parts)}"
                    f" | created={local_stats['observations_created']}"
                    f" updated={local_stats['observations_updated']}"
                    f" skipped={local_stats['skipped']}"
                    + (f" failed={local_stats['memories_failed']}" if local_stats["memories_failed"] else "")
                    + f" | input_tokens=~{input_tokens}"
                    f" | avg={llm_batch_time / max(1, len(llm_batch_local)):.3f}s/memory"
                )

                set_stage(f"consolidation.llm_batch.{batch_num_local}")
                await memory_engine._write_operation_progress(
                    operation_id,
                    stage="consolidating",
                    processed=cum_processed,
                    total=await _progress_total(cum_processed),
                    detail={
                        "observations_created": cum_snapshot["observations_created"],
                        "observations_updated": cum_snapshot["observations_updated"],
                        "observations_merged": cum_snapshot["observations_merged"],
                        "observations_deleted": cum_snapshot["observations_deleted"],
                        "memories_failed": cum_snapshot["memories_failed"],
                    },
                )

                perf.merge_from(batch_perf)

            except BaseException:
                # The witness row was never committed, so this batch's writes are invisible;
                # discard the write-group rather than leaving it pending for the recovery sweep.
                # Kept OUTSIDE decide(commit=True): once the witness has committed, the batch's
                # fate is decided and an abort here would discard durable writes.
                try:
                    await _txn_provider.decide_txn(_batch_txn, commit=False)
                except Exception:
                    logger.warning(
                        f"[CONSOLIDATION] bank={bank_id} failed to abort write-group for"
                        f" llm_batch #{batch_num_local}; recovery sweep will resolve it",
                        exc_info=True,
                    )
                raise

            # The Phase-B transaction committed (witness + marks + observations); publish the group.
            await _txn_provider.decide_txn(_batch_txn, commit=True)

            return _BatchDeltas(stats=local_stats, tags=local_tags, cancelled=cancelled_local)

        # Number every batch up front so log line numbering is deterministic
        # regardless of dispatch order under parallelism. Each group keeps its own
        # (batch, number) list so it can be processed as one serial unit.
        numbered_groups: list[list[tuple[list[dict[str, Any]], int]]] = []
        for batches in grouped_batches:
            numbered: list[tuple[list[dict[str, Any]], int]] = []
            for b in batches:
                llm_batch_num += 1
                numbered.append((b, llm_batch_num))
            numbered_groups.append(numbered)

        async def _process_tag_group(
            group_batches: list[tuple[list[dict[str, Any]], int]],
        ) -> list[_BatchDeltas]:
            # Batches within a group share a tag set and observation scope, so
            # they MUST run serially. Stop early if the op was cancelled mid-group.
            deltas: list[_BatchDeltas] = []
            for b, n in group_batches:
                d = await _process_one_llm_batch(b, n)
                deltas.append(d)
                if d.cancelled:
                    break
            return deltas

        llm_parallelism = max(1, config.consolidation_llm_parallelism)

        if llm_parallelism > 1 and len(numbered_groups) > 1:
            sem = asyncio.Semaphore(llm_parallelism)
            # Per-scope async locks shared across all parallel groups in this
            # fetch iteration. Each group acquires locks for every scope it will
            # write to, in _scope_sort_key order (deadlock-free). Groups with
            # disjoint scope sets never contend; any overlap serialises on the
            # overlapping scopes — covering combined / per_tag / all_combinations
            # / explicit-list modes uniformly without operator opt-in.
            scope_locks: defaultdict[frozenset[str], asyncio.Lock] = defaultdict(asyncio.Lock)

            async def _run_group(
                group_batches: list[tuple[list[dict[str, Any]], int]],
                scopes: list[frozenset[str]],
            ) -> list[_BatchDeltas]:
                async with sem:
                    async with AsyncExitStack() as stack:
                        for s in scopes:
                            await stack.enter_async_context(scope_locks[s])
                        return await _process_tag_group(group_batches)

            group_results = await _gather_or_cancel([_run_group(g, s) for g, s in zip(numbered_groups, group_scopes)])
            batch_results: list[_BatchDeltas] = [d for gd in group_results for d in gd]
            any_cancelled = any(d.cancelled for d in batch_results)
        else:
            batch_results = []
            any_cancelled = False
            for g in numbered_groups:
                group_deltas = await _process_tag_group(g)
                batch_results.extend(group_deltas)
                if any(d.cancelled for d in group_deltas):
                    any_cancelled = True
                    break

        # Merge per-batch deltas into outer state — serial, post-dispatch, so
        # concurrent batches cannot race on the shared counters / tag set.
        for d in batch_results:
            for k, v in d.stats.items():
                stats[k] = stats.get(k, 0) + v
            consolidated_tags.update(d.tags)
            retry_exhausted.update(d.retry_exhausted_ids)

        if any_cancelled:
            return {"status": "cancelled", "bank_id": bank_id, **stats}

        # Update round budget after processing this DB fetch batch
        if round_limit_enabled:
            round_remaining -= len(memories)
            if round_remaining <= 0:
                hit_round_limit = True
                break

    # Re-submit consolidation if we hit the round limit and there's likely more work.
    # Any failure here must propagate: swallowing it (the prior behavior) leaves the
    # bank with backlog and no queued work — silently stuck — because the outer op
    # gets marked completed in the success path. Letting the exception bubble up to
    # execute_task's retry handler means the op is retried with backoff; on retry the
    # consolidator skips already-consolidated rows via the consolidated_at filter and
    # picks up the remainder. Issue #1842.
    # The affected-tag union for the whole round-limited chain. Refresh fires once, when
    # the backlog has fully drained (the final round), not once per round — a model's
    # memories can straddle rounds, and gating on the final round alone (the prior
    # behaviour) dropped every model consolidated earlier because the final round's tags
    # no longer named them (#3411). The union is durable: each batch writes its tags into
    # the op's ``task_payload`` inside the batch's own witness txn (crash-safe), and the
    # re-queue threads the accumulated set forward to the next round. Prefer that durable
    # value; fall back to the in-memory union when there is no backing op (a direct
    # ``run_consolidation_job`` call, e.g. in tests).
    all_refresh_tags = set(pending_refresh_tags or []) | consolidated_tags
    if operation_id:
        all_refresh_tags |= await _read_pending_refresh_tags(pool, operation_id)

    if hit_round_limit:
        remaining = total_count - stats["memories_processed"]
        logger.info(
            f"[CONSOLIDATION] bank={bank_id} hit round limit of {max_memories_per_round} memories,"
            f" ~{remaining} remaining. Re-queuing consolidation."
        )
        await memory_engine.submit_async_consolidation(
            bank_id=bank_id,
            request_context=request_context,
            observation_scopes=observation_scopes,
            pending_refresh_tags=sorted(all_refresh_tags) or None,
        )

    # Build summary
    perf.log(
        f"[3] Results: {stats['memories_processed']} memories -> "
        f"{stats['actions_executed']} actions "
        f"({stats['observations_created']} created, "
        f"{stats['observations_updated']} updated, "
        f"{stats['observations_merged']} merged, "
        f"{stats['skipped']} skipped)"
    )

    # Add timing breakdown. Each phase is recorded once per call, so the count
    # disambiguates a single slow call from many fast calls — important for
    # operators triaging "the recall phase took 15s" log lines, where the
    # total is the sum of many serial sub-calls rather than one slow query.
    def _fmt(key: str) -> str:
        total = perf.timings[key]
        count = perf.timing_counts.get(key, 0)
        if count > 1:
            avg_ms = total * 1000.0 / count
            return f"{key}={total:.3f}s ({count} calls, avg={avg_ms:.0f}ms)"
        return f"{key}={total:.3f}s"

    timing_parts = []
    for key in ("recall", "llm", "embedding", "db_write"):
        if key in perf.timings:
            timing_parts.append(_fmt(key))

    if perf.llm_calls > 0:
        timing_parts.append(f"avg_obs={perf.total_obs_in_context / perf.llm_calls:.1f}")
        timing_parts.append(f"avg_prompt_tokens=~{perf.total_prompt_chars / perf.llm_calls / 4:.0f}")

    if timing_parts:
        perf.log(f"[4] Timing breakdown: {', '.join(timing_parts)}")

    # Trigger mental-model refreshes once, when the chain has fully drained. On a
    # round-limited round we skip and carry the affected tags forward (above); the
    # final round flushes the accumulated union, so a model whose memories were
    # consolidated in ANY round is refreshed exactly once — deduplicated, not dropped
    # (#3411). Each model is still refreshed at most once per drain: a strict tagged
    # model appears once in the trigger's candidate query regardless of how many rounds
    # its tag spanned.
    if hit_round_limit:
        stats["mental_models_refreshed"] = 0
        logger.info(
            f"[CONSOLIDATION] bank={bank_id} deferring mental model refresh to the final round "
            f"(round limit hit; carrying {len(all_refresh_tags)} tags forward)"
        )
    else:
        set_stage("consolidation.refreshing_mental_models")
        await memory_engine._write_operation_progress(
            operation_id,
            stage="refreshing_mental_models",
            processed=stats["memories_processed"],
            total=await _progress_total(stats["memories_processed"]),
        )
        # SECURITY: Only refresh mental models whose scope covers what was consolidated
        mental_models_refreshed = await _trigger_mental_model_refreshes(
            memory_engine=memory_engine,
            bank_id=bank_id,
            request_context=request_context,
            consolidated_tags=sorted(all_refresh_tags) or None,
            perf=perf,
        )
        stats["mental_models_refreshed"] = mental_models_refreshed

    perf.flush()

    return {"status": "completed", "bank_id": bank_id, **stats}


# SQL predicate: "this mental model's refresh scope can contain untagged memories".
#
# A model's scope is NOT its ``tags`` column — it is whatever
# ``_resolve_refresh_tag_filtering`` resolves, and both the refresh and the staleness
# check use that. Three cases reach untagged memories:
#   - no tags at all             -> no tag constraint, every bank memory is in scope
#   - tags_match "any" / "all"   -> non-strict, the clause ORs in untagged rows
#   - trigger.tag_groups         -> overrides the tags column entirely, so the column
#                                   says nothing about what the model can see
# A tagged model left on the default (``all_strict``) is correctly excluded: strict
# matching drops untagged rows, so an untagged-only consolidation cannot make it stale.
# Gating on the tags column alone starved the first two cases (#3053).
_MM_SCOPE_REACHES_UNTAGGED = (
    "((tags IS NULL OR tags = '{}') OR (trigger->>'tags_match') IN ('any', 'all') OR trigger ? 'tag_groups')"
)


async def _trigger_mental_model_refreshes(
    memory_engine: "MemoryEngine",
    bank_id: str,
    request_context: "RequestContext",
    consolidated_tags: list[str] | None = None,
    perf: ConsolidationPerfLog | None = None,
) -> int:
    """
    Trigger refreshes for mental models with refresh_after_consolidation=true.

    SECURITY: Only triggers refresh for mental models whose refresh scope can contain
    what this consolidation touched, preventing unnecessary refreshes across security
    boundaries.

    Args:
        memory_engine: MemoryEngine instance
        bank_id: Bank identifier
        request_context: Request context for authentication
        consolidated_tags: Tags of the memories that were consolidated. None means only
            untagged memories were consolidated (or nothing was), so only models whose
            scope reaches untagged memories are candidates.
        perf: Performance logging

    Returns:
        Number of mental models scheduled for refresh
    """
    pool = memory_engine._backend

    # Find mental models with refresh_after_consolidation=true that are actually stale.
    # The tag predicate on the SELECT is a cheap prefilter that skips models this
    # consolidation cannot have affected; compute_mental_model_is_stale then verifies
    # against the model's *resolved* scope that new memories really were ingested since
    # its last refresh.
    async with acquire_with_retry(pool) as conn:
        if consolidated_tags:
            candidates = await conn.fetch(
                f"""
                SELECT id, name, tags, last_refreshed_at, trigger
                FROM {fq_table("mental_models")}
                WHERE bank_id = $1
                  AND (trigger->>'refresh_after_consolidation')::boolean = true
                  AND (
                    (tags IS NOT NULL AND tags != '{{}}' AND tags && $2::varchar[])
                    OR {_MM_SCOPE_REACHES_UNTAGGED}
                  )
                """,
                bank_id,
                consolidated_tags,
            )
        else:
            candidates = await conn.fetch(
                f"""
                SELECT id, name, tags, last_refreshed_at, trigger
                FROM {fq_table("mental_models")}
                WHERE bank_id = $1
                  AND (trigger->>'refresh_after_consolidation')::boolean = true
                  AND {_MM_SCOPE_REACHES_UNTAGGED}
                """,
                bank_id,
            )

        rows = []
        for candidate in candidates:
            if await memory_engine.compute_mental_model_is_stale(conn, bank_id, candidate):
                rows.append(candidate)

    if not rows:
        return 0

    if perf:
        if consolidated_tags:
            perf.log(
                f"[5] Triggering refresh for {len(rows)} mental models with refresh_after_consolidation=true "
                f"(filtered by tags: {consolidated_tags})"
            )
        else:
            perf.log(f"[5] Triggering refresh for {len(rows)} mental models with refresh_after_consolidation=true")

    # Submit refresh tasks for each mental model
    refreshed_count = 0
    for row in rows:
        mental_model_id = row["id"]
        try:
            # skip_if_in_flight: a consolidation chain fires this every round and
            # overlapping consolidations can run on the same bank, so a model still
            # pending/processing a refresh must not be enqueued a second time (#3411).
            await memory_engine.submit_async_refresh_mental_model(
                bank_id=bank_id,
                mental_model_id=mental_model_id,
                request_context=request_context,
                skip_if_in_flight=True,
            )
            refreshed_count += 1
            logger.info(
                f"[CONSOLIDATION] Triggered refresh for mental model {mental_model_id} "
                f"(name: {row['name']}) in bank {bank_id}"
            )
        except Exception as e:
            logger.warning(f"[CONSOLIDATION] Failed to trigger refresh for mental model {mental_model_id}: {e}")

    return refreshed_count


async def _prepare_memory_batch(
    pool: DatabaseBackend,
    memory_engine: "MemoryEngine",
    llm_config: Any,
    bank_id: str,
    memories: list[dict[str, Any]],
    request_context: "RequestContext",
    perf: ConsolidationPerfLog | None = None,
    config: Any = None,
    obs_tags_override: list[str] | None = None,
) -> _PreparedBatch:
    """Phase A — prepare a batch of memories for one LLM call, off any write connection.

    Runs the slow work — per-fact recall, the unioned observation set, observation-slot
    accounting, the single LLM call, action embeddings, and semantic-dedup adjudication —
    entirely connection-free for writes. Each helper acquires only short-lived *read*
    connections around its own SQL; no bank lock or transaction is held across any slow
    step (design §4.2). Returns a :class:`_PreparedBatch` carrying every write the batch
    intends, plus the pre-computed embeddings and dedup outcomes, so Phase B
    (:func:`_commit_prepared_batch`) performs only bounded reads + CAS writes under the
    bank guard.

    Per-fact security: action execution validates each learning_id against the
    observations that were recalled specifically for that fact, so cross-tag
    updates cannot occur.
    """
    # Map the source memories this batch consumes onto the consolidation trace.
    record_source_memory_ids([str(m["id"]) for m in memories])

    # 1. Parallel recalls — one per fact
    # When obs_tags_override is set, use it as the observation scope for all facts.
    t0 = time.time()
    observation_scope_tags = obs_tags_override if obs_tags_override is not None else None
    recall_tasks = [
        _find_related_observations(
            memory_engine=memory_engine,
            bank_id=bank_id,
            query=m["text"],
            request_context=request_context,
            tags=observation_scope_tags if observation_scope_tags is not None else (m.get("tags") or []),
        )
        for m in memories
    ]
    # A failed recall must fail the batch rather than degrade to "no related
    # observations": proceeding with an empty candidate set would hide an
    # existing twin from the LLM and turn an UPDATE into a duplicate CREATE.
    # The batch's memories stay unconsolidated and are picked up on retry.
    per_fact_recalls = await _gather_or_cancel(recall_tasks)
    if perf:
        perf.record_timing("recall", time.time() - t0)

    # 2. Build per-fact observation sets (keyed by memory ID string) for secure action validation
    per_fact_obs_ids: dict[str, set[str]] = {
        str(memories[i]["id"]): {str(obs.id) for obs in r.results} for i, r in enumerate(per_fact_recalls)
    }

    # Union all observations (deduped by id)
    seen_ids: set[str] = set()
    union_observations: list["MemoryFact"] = []
    union_source_facts: dict[str, "MemoryFact"] = {}
    for recall_result in per_fact_recalls:
        for obs in recall_result.results:
            obs_id = str(obs.id)
            if obs_id not in seen_ids:
                seen_ids.add(obs_id)
                union_observations.append(obs)
        if recall_result.source_facts:
            union_source_facts.update(recall_result.source_facts)

    # Snapshot BEFORE the LLM so the shown observations and stored tokens are the
    # same StoredMemory objects (judge 76f0ac68). Missing rows are dropped.
    union_observations, observation_revisions = await _bind_recalled_observations_to_snapshots(
        pool, bank_id, union_observations
    )
    live_obs_ids = set(observation_revisions)
    per_fact_obs_ids = {mid: (ids & live_obs_ids) for mid, ids in per_fact_obs_ids.items()}

    # Determine effective tag scope for observations.
    # When obs_tags_override is set, use it; otherwise use the memory's own tags.
    if obs_tags_override is not None:
        fact_tags = obs_tags_override
    else:
        # All memories in the batch share the same tag set (enforced by batching)
        fact_tags = memories[0].get("tags") or [] if memories else []

    # 2b. Compute remaining observation slots for this scope (if limit configured).
    # The cap is resolved per-scope: an observation_scope_limits rule may override
    # the bank-wide max_observations_per_scope for scopes matching its tag pattern.
    max_obs = _effective_scope_limit(config, fact_tags)
    remaining_observation_slots: int | None = None
    if max_obs >= 0 and fact_tags:
        # max_obs == 0 means "no new observations": there are no slots regardless
        # of the current count, so skip the count query for that case.
        current_count = 0
        if max_obs > 0:
            async with acquire_with_retry(pool) as count_conn:
                current_count = await _count_observations_for_scope(count_conn, bank_id, fact_tags)
        remaining_observation_slots = max(max_obs - current_count, 0)
        if remaining_observation_slots == 0:
            logger.info(
                f"[CONSOLIDATION] bank={bank_id} scope={fact_tags} at observation limit "
                f"({current_count}/{max_obs}), only updates/deletes allowed"
            )

    # 3. Single LLM call
    t0 = time.time()
    llm_result = await _consolidate_batch_with_llm(
        llm_config=llm_config,
        memories=memories,
        union_observations=union_observations,
        union_source_facts=union_source_facts,
        config=config,
        remaining_observation_slots=remaining_observation_slots,
        max_observations_per_scope=max_obs,
    )
    if perf:
        perf.record_timing("llm", time.time() - t0)
        perf.record_llm_call(llm_result.obs_count, llm_result.prompt_chars)

    dedup_enabled = _dedup_active(config)
    dedup_llm_config = (
        memory_engine._consolidation_llm_config.with_config(config, bank_id=bank_id, operation="consolidation_dedup")
        if dedup_enabled
        else None
    )

    mem_by_id = {str(m["id"]): m for m in memories}

    # Deterministic dedup guard: map the observations the LLM was SHOWN by their
    # normalised text. The model intermittently emits a CREATE whose text is identical
    # to an observation already in its context (over-aggregation / incoherence — it even
    # UPDATEs the twin and creates a sibling). When that happens we drop the duplicate
    # CREATE instead of inserting a redundant row. No extra LLM/embedding cost — the
    # match is exact text against the in-memory set.
    shown_obs_by_text = {_norm_obs_text(o.text): o for o in union_observations}
    # Also collapse a CREATE that reproduces the text of an UPDATE issued in the SAME
    # response (the model occasionally UPDATEs the twin to text X and also CREATEs X).
    update_texts = {_norm_obs_text(u.text) for u in llm_result.updates if u.text}

    # Phase A prepares each action's plan: source aggregation, embedding (slow), and
    # semantic-dedup adjudication (LLM) all happen here — off any write connection — so
    # Phase B holds the bank guard only for bounded reads + CAS writes.
    prepared_deletes: list[_PreparedDelete] = []
    prepared_updates: list[_PreparedUpdate] = []
    prepared_creates: list[_PreparedCreate] = []

    for delete in llm_result.deletes:
        # Security: the observation must be present in the unioned recall.
        if not any(str(obs.id) == delete.observation_id for obs in union_observations):
            logger.debug(
                f"Batch consolidation: rejected delete — observation {delete.observation_id} not in unioned recall"
            )
            continue
        prepared_deletes.append(
            _PreparedDelete(
                delete=delete,
                phase_a_revision=observation_revisions.get(str(delete.observation_id), ""),
            )
        )

    for update in llm_result.updates:
        source_mems = [mem_by_id[fid] for fid in update.source_fact_ids if fid in mem_by_id]
        if not source_mems:
            continue
        # Security: the observation must have been recalled for at least one of the source facts
        if not any(update.observation_id in per_fact_obs_ids.get(str(m["id"]), set()) for m in source_mems):
            logger.debug(
                f"Batch consolidation: rejected update — observation {update.observation_id} "
                f"not in any source fact's recall"
            )
            continue
        agg = _aggregate_source_fields(source_mems, tags=fact_tags)
        embedding_str: str | None = None
        dedup_outcome: "_DedupOutcome | None" = None
        if dedup_enabled:
            embeddings = await embedding_utils.generate_embeddings_batch(memory_engine.embeddings, [update.text])
            embedding_str = str(embeddings[0]) if embeddings else None
            if embedding_str is not None:
                dedup_outcome = await _dedup_adjudicate(
                    pool,
                    memory_engine,
                    bank_id,
                    config,
                    dedup_llm_config,
                    update.text,
                    embedding_str,
                    agg.tags,
                    exclude_id=update.observation_id,
                )
        prepared_updates.append(
            _PreparedUpdate(
                update=update,
                source_mems=source_mems,
                agg=agg,
                embedding_str=embedding_str,
                dedup_outcome=dedup_outcome,
                phase_a_revision=observation_revisions.get(str(update.observation_id), ""),
            )
        )

    for create in llm_result.creates:
        source_mems = [mem_by_id[fid] for fid in create.source_fact_ids if fid in mem_by_id]
        if not source_mems:
            continue
        agg = _aggregate_source_fields(source_mems, tags=fact_tags)
        create_source_ids = [m["id"] for m in source_mems]

        # Reconcile against observations shown to the LLM: an exact-text match means
        # this CREATE reproduces verbatim an observation the model already had in context.
        duplicate_of = _duplicate_create_target(create.text, shown_obs_by_text, update_texts)
        if duplicate_of is not None:
            logger.warning(
                "[CONSOLIDATION] dropped duplicate observation CREATE — verbatim match of %s; llm_reason=%r",
                duplicate_of,
                create.reason or "(none given)",
            )
            continue

        dedup_outcome: "_DedupOutcome | None" = None
        embedding_str: str | None = None
        if dedup_enabled:
            # Precompute the CREATE embedding in Phase A (design §4.2 — no embedder
            # under the bank lock). The dedup adjudication uses it as its probe vector.
            embeddings = await embedding_utils.generate_embeddings_batch(memory_engine.embeddings, [create.text])
            embedding_str = str(embeddings[0]) if embeddings else None
            dedup_outcome = await _dedup_adjudicate(
                pool,
                memory_engine,
                bank_id,
                config,
                dedup_llm_config,
                create.text,
                embedding_str,
                agg.tags,
                exclude_id=None,
            )

        cand_revs = dict(dedup_outcome.candidate_revisions) if dedup_outcome is not None else {}
        twin_rev = None
        if dedup_outcome is not None and dedup_outcome.best_id is not None:
            twin_rev = cand_revs.get(str(dedup_outcome.best_id))
        prepared_creates.append(
            _PreparedCreate(
                create=create,
                source_mems=source_mems,
                agg=agg,
                create_source_ids=create_source_ids,
                dedup_outcome=dedup_outcome,
                embedding_str=embedding_str,
                candidate_ids=set(dedup_outcome.candidate_ids) if dedup_outcome is not None else set(),
                phase_a_target_revision=twin_rev,
                candidate_revisions=cand_revs,
            )
        )

    return _PreparedBatch(
        memories=memories,
        per_fact_obs_ids=per_fact_obs_ids,
        union_observations=union_observations,
        llm_result=llm_result,
        fact_tags=fact_tags,
        deletes=prepared_deletes,
        updates=prepared_updates,
        creates=prepared_creates,
        dedup_enabled=dedup_enabled,
        dedup_llm_config=dedup_llm_config,
        source_snapshots={str(m["id"]): _source_fingerprint(m) for m in memories},
        observation_revisions=observation_revisions,
    )


async def _commit_prepared_batch(
    prepared: _PreparedBatch,
    pool: DatabaseBackend,
    memory_engine: "MemoryEngine",
    bank_id: str,
    config: Any = None,
    perf: ConsolidationPerfLog | None = None,
    txn=None,
    conn=None,
) -> tuple[list[dict[str, Any]], int, set[str]]:
    """Phase B — commit a prepared batch's writes under CAS on the caller-owned connection.

    Executes deletes first (freeing observation slots before creates consume them), then
    updates (+ fold), then creates (+ fold), each through the store CAS seam with fresh
    validation inside ``conn``'s write-group. When ``conn`` is given (parallel Phase-B mode)
    all writes share one caller-owned transaction and no nested transaction is opened; when
    omitted (serial path) each write opens its own short-lived transaction via ``pool``.

    Returns ``(results, deleted_count, stale_memory_ids)`` where ``results`` is one dict per
    memory in batch order and ``stale_memory_ids`` are the source memories whose prepared
    writes were dropped with zero writes because fresh validation under the guard found them
    consumed/mutated/duplicated (design §6.1). Those memories must stay unconsolidated so a
    bounded reprepare can fold them into the survivor.
    """
    memories = prepared.memories
    per_memory_created: set[str] = set()
    per_memory_updated: set[str] = set()
    stale_memory_ids: set[str] = set()
    deleted_count = 0

    async def _write_body(wconn) -> None:
        nonlocal deleted_count

        for pdel in prepared.deletes:
            await _execute_delete_action(
                conn=wconn,
                bank_id=bank_id,
                observation_id=pdel.delete.observation_id,
                txn=txn,
                expected_revision=pdel.phase_a_revision,
            )
            deleted_count += 1

        for pupd in prepared.updates:
            update = pupd.update
            updated_emb_str = await _execute_update_action(
                pool=pool,
                memory_engine=memory_engine,
                bank_id=bank_id,
                source_memory_ids=[m["id"] for m in pupd.source_mems],
                observation_id=update.observation_id,
                new_text=update.text,
                observations=prepared.union_observations,
                source_fact_tags=pupd.agg.tags,
                source_occurred_start=pupd.agg.occurred_start,
                source_occurred_end=pupd.agg.occurred_end,
                source_mentioned_at=pupd.agg.mentioned_at,
                perf=perf,
                txn=txn,
                conn=wconn,
                precomputed_embedding=pupd.embedding_str,
                expected_revision=pupd.phase_a_revision,
            )
            for m in pupd.source_mems:
                per_memory_updated.add(str(m["id"]))
            # Reconcile the rewritten observation against its neighbours: re-embed may have
            # drifted it into a near-twin of another existing observation. updated emb is None
            # when skipped; nothing to reconcile then.
            if prepared.dedup_enabled and updated_emb_str is not None:
                await _dedup_reconcile_update(
                    pool,
                    memory_engine,
                    bank_id,
                    config,
                    prepared.dedup_llm_config,
                    update.observation_id,
                    update.text,
                    updated_emb_str,
                    pupd.agg.tags,
                    txn=txn,
                    conn=wconn,
                    outcome=pupd.dedup_outcome,
                )

        for pcreate in prepared.creates:
            create = pcreate.create

            # Fresh source-ID + candidate-set revalidation under the bank guard (design §4.2
            # step 3-4, §6.1). A writer whose sources were consumed or mutated during the
            # connection-free LLM window must roll back with zero writes rather than create a
            # duplicate observation. This is what closes the empty-snapshot CREATE/CREATE race.
            stale_reason = await _fresh_source_validation(
                conn=wconn,
                bank_id=bank_id,
                source_ids=pcreate.create_source_ids or [],
                expected_fingerprints=prepared.source_snapshots,
            )
            if stale_reason != "ok":
                logger.debug(
                    f"[CONSOLIDATION] bank={bank_id} CREATE stale under guard ({stale_reason}); "
                    f"dropping with zero writes (duplicate/twin protection §6.1)"
                )
                # The memory is NOT marked consolidated/failed: it stays eligible for a
                # bounded reprepare that folds it into the survivor observation.
                stale_memory_ids.update(str(m["id"]) for m in pcreate.source_mems)
                continue

            if (
                pcreate.dedup_outcome is not None
                and pcreate.dedup_outcome.should_merge
                and pcreate.dedup_outcome.best_id is not None
            ):
                merged_into = await _dedup_reconcile_create(
                    pool,
                    memory_engine,
                    bank_id,
                    config,
                    prepared.dedup_llm_config,
                    create.text or "",
                    pcreate.create_source_ids or [],
                    pcreate.agg.tags or [],
                    txn=txn,
                    conn=wconn,
                    outcome=pcreate.dedup_outcome,
                    expected_revision=pcreate.phase_a_target_revision,
                )
                if merged_into is not None:
                    logger.info(
                        "[CONSOLIDATION] dedup-merged observation CREATE into %s (cosine>=%.2f)",
                        merged_into[:8],
                        config.consolidation_dedup_threshold if config else 0.9,
                    )
                    for m in pcreate.source_mems:
                        per_memory_created.add(str(m["id"]))
                    continue

            action = await _execute_create_action(
                pool=pool,
                memory_engine=memory_engine,
                bank_id=bank_id,
                source_memory_ids=pcreate.create_source_ids or [],
                text=create.text or "",
                source_fact_tags=pcreate.agg.tags or [],
                event_date=pcreate.agg.event_date,
                occurred_start=pcreate.agg.occurred_start,
                occurred_end=pcreate.agg.occurred_end,
                mentioned_at=pcreate.agg.mentioned_at,
                perf=perf,
                txn=txn,
                conn=wconn,
                precomputed_embedding=pcreate.embedding_str,
            )
            if action == "created":
                for m in pcreate.source_mems:
                    per_memory_created.add(str(m["id"]))

    async with _write_group(pool, conn) as wconn:
        await _write_body(wconn)

    # Build per-memory result dicts for the stats tracker in the outer loop.
    results: list[dict[str, Any]] = []
    for m in memories:
        mid = str(m["id"])
        created = mid in per_memory_created
        updated = mid in per_memory_updated
        if created and updated:
            results.append({"action": "multiple", "created": 1, "updated": 1, "merged": 0, "total_actions": 2})
        elif created:
            results.append({"action": "created"})
        elif updated:
            results.append({"action": "updated"})
        elif mid in stale_memory_ids:
            results.append({"action": "skipped", "reason": "stale_under_guard", "retryable": True})
        else:
            results.append({"action": "skipped", "reason": "no_durable_knowledge"})

    return results, deleted_count, stale_memory_ids


async def _process_memory_batch(
    pool: DatabaseBackend,
    memory_engine: "MemoryEngine",
    llm_config: Any,
    bank_id: str,
    memories: list[dict[str, Any]],
    request_context: "RequestContext",
    perf: ConsolidationPerfLog | None = None,
    config: Any = None,
    obs_tags_override: list[str] | None = None,
    txn=None,
    conn=None,
) -> tuple[list[dict[str, Any]], int, bool]:
    """Process a batch of memories in a single LLM call.

    Composes :func:`_prepare_memory_batch` (Phase A — slow work, off write connection)
    and :func:`_commit_prepared_batch` (Phase B — CAS writes on the caller-owned connection).

    Steps:
    1. Parallel recalls — one per fact (read-only; safe to parallelise)
    2. Union of retrieved observations across the batch (deduped by id)
    3. Single LLM call with all N facts + unioned observations
    4. Sequential action execution (writes remain serial for consistency)
    5. Returns one result dict per memory, in the same order as `memories`

    ``obs_tags_override`` uses these tags for observation recall and create/update instead
    of the memory's own tags (multi-pass consolidation). ``conn`` when given runs Phase B on
    the caller-owned connection (parallel mode); when omitted each write opens its own short
    transaction (serial path) exactly as before.
    """
    prepared = await _prepare_memory_batch(
        pool=pool,
        memory_engine=memory_engine,
        llm_config=llm_config,
        bank_id=bank_id,
        memories=memories,
        request_context=request_context,
        perf=perf,
        config=config,
        obs_tags_override=obs_tags_override,
    )
    results, deleted_count, _stale = await _commit_prepared_batch(
        prepared=prepared,
        pool=pool,
        memory_engine=memory_engine,
        bank_id=bank_id,
        config=config,
        perf=perf,
        txn=txn,
        conn=conn,
    )
    return results, deleted_count, prepared.llm_result.failed


def _min_date(dates: "Any") -> "datetime | None":
    """Return the minimum non-None datetime from an iterable."""
    return min((d for d in dates if d is not None), default=None)


def _max_date(dates: "Any") -> "datetime | None":
    """Return the maximum non-None datetime from an iterable."""
    return max((d for d in dates if d is not None), default=None)


@dataclass(frozen=True)
class _ObservationHistorySnapshot:
    """Pre-update state of an observation, persisted as the ``content`` JSON blob
    of one observation_history row.

    Temporal fields are the ISO strings carried on MemoryFact; new_source_memory_ids
    are the ids added by the update.
    """

    previous_text: str | None
    previous_tags: list[str]
    previous_occurred_start: str | None
    previous_occurred_end: str | None
    previous_mentioned_at: str | None
    new_source_memory_ids: list[str]


async def _append_observation_history(
    conn: "Connection",
    bank_id: str,
    observation_id: str,
    snapshot: _ObservationHistorySnapshot,
    max_entries: int,
) -> None:
    """Insert one pre-update snapshot into ``observation_history``, then delete the
    oldest rows beyond ``max_entries`` for this observation.

    The snapshot is stored as a single JSONB ``content`` blob (per-row, so it stays
    small). Bounding by row count keeps a frequently-reinforced observation's
    history from growing without bound.
    """
    obs_uuid = uuid.UUID(observation_id)
    try:
        await conn.execute(
            f"""
        INSERT INTO {fq_table("observation_history")} (observation_id, bank_id, content, changed_at)
        VALUES ($1, $2, $3::jsonb, now())
        """,
            obs_uuid,
            bank_id,
            json.dumps(asdict(snapshot)),
        )
    except asyncpg.exceptions.ForeignKeyViolationError:
        logger.warning(
            f"FK violation writing observation_history for {observation_id}: "
            "observation was removed before history could be written (race with parallel consolidation). Skipping."
        )
        return
    if max_entries and max_entries > 0:
        await conn.execute(
            f"""
            DELETE FROM {fq_table("observation_history")}
            WHERE observation_id = $1
              AND id NOT IN (
                  SELECT id FROM {fq_table("observation_history")}
                  WHERE observation_id = $1
                  ORDER BY changed_at DESC, id DESC
                  LIMIT $2
              )
            """,
            obs_uuid,
            max_entries,
        )


async def _execute_update_action(
    pool: DatabaseBackend,
    memory_engine: "MemoryEngine",
    bank_id: str,
    source_memory_ids: list[uuid.UUID],
    observation_id: str,
    new_text: str,
    observations: list["MemoryFact"],
    source_fact_tags: list[str] | None = None,
    source_occurred_start: datetime | None = None,
    source_occurred_end: datetime | None = None,
    source_mentioned_at: datetime | None = None,
    perf: ConsolidationPerfLog | None = None,
    txn=None,
    conn=None,
    precomputed_embedding: str | None = None,
    expected_revision: str | None = None,
) -> str | None:
    """
    Update an existing observation.

    Extends source_memory_ids with all contributing memories, updates temporal fields
    (LEAST for occurred_start, GREATEST for occurred_end / mentioned_at), and merges tags.

    The embedding is computed off-connection (a slow embedder must never pin a pooled
    connection); the liveness check + UPDATE + history + observation_sources sync then run
    in one short transaction so they commit atomically.

    ``precomputed_embedding``: Phase-B caller-owned mode passes the embedding computed
    in Phase A (design §4.2 — no embedder under the bank lock). When omitted (serial
    path, or a non-parallel caller) the embedding is computed here off-connection exactly
    as before.

    Under the bank guard (Phase B) the target observation is snapshotted fresh and the row
    update runs through the store CAS seam (``cas_update_memory``) gated on the expected
    revision token; a concurrently-mutated/deleted target raises ``_BatchStaleError`` so the
    whole original batch rolls back with zero writes (judge blocker 5, Ruling 1).

    Returns the observation's freshly-computed embedding (pgvector literal) so the caller can
    run UPDATE-path dedup without re-embedding, or None when the update was skipped.

    ``conn``: when provided (Phase-B caller-owned mode) all writes run on it inside the
    caller's write-group transaction; no transaction is opened here and no connection is
    acquired. When omitted (serial path) a short-lived connection + transaction are opened
    exactly as before.
    """
    model = next((m for m in observations if str(m.id) == observation_id), None)
    if not model:
        logger.debug(f"Update skipped: observation {observation_id} not found in recall results")
        return None

    from ...config import get_config

    # Preflight (non-locking, separate short-lived conn): if every source memory is already
    # gone, skip BEFORE the slow embed — restores the pre-refactor short-circuit so a no-op
    # update doesn't embed and a failing embedder doesn't raise where it used to skip.
    # Skipped in caller-owned mode: Phase B re-validates source liveness fresh under the bank
    # guard inside ``_write_group``, so a second preflight connection would only add lock traffic.
    if conn is None:
        async with acquire_with_retry(pool) as c:
            if not await _any_live_source_memory(c, bank_id, source_memory_ids):
                logger.debug(
                    f"Update skipped: all {len(source_memory_ids)} source memories for observation "
                    f"{observation_id} were deleted before embedding"
                )
                return None

    # Embed off-connection: the new text is known up front and does not depend on
    # any DB state, so the (slow) embedder runs before we touch the pool. In Phase-B
    # caller-owned mode the embedding was already computed in Phase A and is passed in —
    # never run the slow embedder while holding the bank lock.
    t0 = time.time()
    if precomputed_embedding is not None:
        embedding_str = precomputed_embedding
        if perf:
            perf.record_timing("embedding", 0.0)  # measured in Phase A; do not double-count under lock
    else:
        embeddings = await embedding_utils.generate_embeddings_batch(memory_engine.embeddings, [new_text])
        embedding_str = str(embeddings[0]) if embeddings else None
        if perf:
            perf.record_timing("embedding", time.time() - t0)

    config = get_config()
    store = get_memories()

    async with _write_group(pool, conn) as conn:
        # FOR SHARE liveness + the write share one tiny transaction so a concurrent
        # delete cannot remove a source row between the check and the UPDATE.
        live_source_memory_ids = await _filter_live_source_memories(conn, bank_id, source_memory_ids)
        if not live_source_memory_ids:
            logger.debug(
                f"Update skipped: all {len(source_memory_ids)} source memories for observation "
                f"{observation_id} were deleted concurrently"
            )
            return None
        live_ids = live_source_memory_ids

        history_entry = _ObservationHistorySnapshot(
            previous_text=model.text,
            previous_tags=list(model.tags or []),
            previous_occurred_start=model.occurred_start,
            previous_occurred_end=model.occurred_end,
            previous_mentioned_at=model.mentioned_at,
            new_source_memory_ids=[str(mid) for mid in live_ids],
        )

        source_ids = list(model.source_fact_ids or []) + live_ids

        # SECURITY: Merge source fact's tags into existing observation tags so all contributors can see it
        existing_tags = set(model.tags or [])
        source_tags = set(source_fact_tags or [])
        merged_tags = list(existing_tags | source_tags)

        t0 = time.time()
        if store.writes_memory_rows_in_sql_for(bank_id):
            # Blocker 5 (judge 5f7f900d): the target observation must be mutated through the
            # store CAS seam, never by an unconditional row update. Snapshot it fresh under the
            # caller's transaction (bank -> observation lock order), then apply via
            # ``cas_update_memory`` gated on the expected revision token. A concurrently-
            # mutated/deleted target returns STALE/MISSING -> raise ``_BatchStaleError`` so the
            # whole original batch rolls back with zero writes (Ruling 1).
            snaps = await store.snapshot_memories(
                conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[observation_id]
            )
            if not snaps:
                logger.debug(
                    f"Update aborted: observation {observation_id} no longer exists (deleted/invalidated concurrently)"
                )
                raise _BatchStaleError(f"update_target_missing:{observation_id}")
            fresh = snaps[0].memory
            if not expected_revision:
                raise _BatchStaleError(f"update_target_missing_phase_a_revision:{observation_id}")
            # Merge against the FRESH row (authoritative under the lock), not the Phase-A
            # ``model`` — LEAST/GREATEST semantics over existing + new values.
            fresh_source_ids = list(fresh.source_memory_ids or [])
            merged_source_ids = [str(s) for s in fresh_source_ids] + [str(mid) for mid in live_ids]
            merged_source_ids = list(dict.fromkeys(merged_source_ids))
            fresh_tags = set(fresh.tags or [])
            merged_tags = list(fresh_tags | source_tags)
            patch = MemoryPatch(
                unit_id=observation_id,
                text=new_text,
                embedding=embedding_str,
                tags=merged_tags,
                proof_count_delta=len(merged_source_ids) - len(fresh_source_ids),
                occurred_start=_merge_min(fresh.occurred_start, source_occurred_start),
                occurred_end=_merge_max(fresh.occurred_end, source_occurred_end),
                mentioned_at=_merge_max(fresh.mentioned_at, source_mentioned_at),
                source_memory_ids=merged_source_ids,
                search_vector=(_native_search_vector_update(config, "{text_param}") or None),
            )
            outcome = await store.cas_update_memory(
                conn=conn,
                fq_table=fq_table,
                bank_id=bank_id,
                unit_id=observation_id,
                expected_revision=expected_revision,
                patch=patch,
            )
            if outcome == CASOutcome.STALE:
                logger.warning(
                    f"Update aborted: observation {observation_id} mutated concurrently "
                    "(CAS STALE); aborting whole batch with zero writes"
                )
                raise _BatchStaleError(f"update_target_stale:{observation_id}")
            if outcome == CASOutcome.MISSING:
                logger.warning(
                    f"Update aborted: observation {observation_id} deleted concurrently "
                    "(CAS MISSING); aborting whole batch with zero writes"
                )
                raise _BatchStaleError(f"update_target_missing:{observation_id}")
            source_ids = merged_source_ids
        else:
            # Upsert overwrites the whole observation, so start from its current state (fetched
            # from the store) and apply the same merge the SQL does — LEAST/GREATEST on the times
            # — while preserving fields the update never touches (event_date, created_at).
            current = await store.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[observation_id])
            cur = current[0] if current else None
            await store.upsert_observation(
                conn=conn,
                bank_id=bank_id,
                txn=txn,
                record=FactRecord(
                    unit_id=observation_id,
                    text=new_text,
                    embedding=embedding_str,
                    fact_type="observation",
                    tags=merged_tags,
                    proof_count=len(source_ids),
                    source_memory_ids=[str(s) for s in source_ids],
                    event_date=cur.event_date if cur else None,
                    occurred_start=_merge_min(model.occurred_start, source_occurred_start),
                    occurred_end=_merge_max(model.occurred_end, source_occurred_end),
                    mentioned_at=_merge_max(model.mentioned_at, source_mentioned_at),
                    created_at=cur.created_at if cur else None,
                ),
            )

        # Record the pre-update snapshot in the dedicated observation_history table
        # (one row per change), then trim to the configured cap. History lived in a
        # single unbounded JSONB column before; an often-reinforced observation grew
        # it until it crossed Postgres's 256MB jsonb limit and got stuck.
        if config.enable_observation_history:
            await _append_observation_history(
                conn, bank_id, observation_id, history_entry, config.observation_history_max_entries
            )

        # Sync observation_sources junction table (Oracle only — PG uses native array ops).
        if memory_engine._backend.ops.uses_observation_sources_table:
            obs_uuid = uuid.UUID(observation_id)
            await conn.execute(
                f"DELETE FROM {fq_table('observation_sources')} WHERE observation_id = $1",
                obs_uuid,
            )
            if source_ids:
                await conn.executemany(
                    f"""
                    INSERT INTO {fq_table("observation_sources")} (observation_id, source_id)
                    VALUES ($1, $2)
                    ON CONFLICT (observation_id, source_id) DO NOTHING
                    """,
                    [(obs_uuid, sid) for sid in dict.fromkeys(source_ids)],
                )

        if perf:
            perf.record_timing("db_write", time.time() - t0)

    # Map the updated observation onto the consolidation trace as a produced memory.
    record_created_memory_ids([observation_id])
    logger.debug(f"Updated observation {observation_id} from {len(source_memory_ids)} source memories")
    return embedding_str


async def _execute_create_action(
    pool: DatabaseBackend,
    memory_engine: "MemoryEngine",
    bank_id: str,
    source_memory_ids: list[uuid.UUID],
    text: str,
    source_fact_tags: list[str] | None = None,
    event_date: datetime | None = None,
    occurred_start: datetime | None = None,
    occurred_end: datetime | None = None,
    mentioned_at: datetime | None = None,
    perf: ConsolidationPerfLog | None = None,
    txn=None,
    conn=None,
    precomputed_embedding: str | None = None,
) -> str:
    """
    Create a new observation from one or more source memories.

    Tags are inherited from the source facts (determined algorithmically, not by LLM)
    to maintain visibility scope. Returns the write action ("created" or "skipped").

    ``conn``: when provided (Phase-B caller-owned mode) it is passed through to the
    create write so the observation lands inside the caller's write-group transaction.

    ``precomputed_embedding``: Phase-B caller-owned mode passes the embedding computed
    in Phase A (design §4.2 — no embedder under the bank lock). When omitted (serial
    path, or a non-parallel caller) the embedding is computed here off-connection exactly
    as before.
    """
    created = await _create_observation_directly(
        pool=pool,
        memory_engine=memory_engine,
        bank_id=bank_id,
        source_memory_ids=source_memory_ids,
        observation_text=text,
        tags=source_fact_tags or [],
        event_date=event_date,
        occurred_start=occurred_start,
        occurred_end=occurred_end,
        mentioned_at=mentioned_at,
        perf=perf,
        txn=txn,
        conn=conn,
        precomputed_embedding=precomputed_embedding,
    )
    # Map the new observation onto the consolidation trace as a produced memory.
    new_id = created.get("observation_id")
    if new_id:
        record_created_memory_ids([new_id])
    logger.debug(f"Created observation from {len(source_memory_ids)} source memories")
    return created["action"]


async def _delete_observation_history(
    conn: "Connection",
    bank_id: str,
    observation_id: str,
) -> None:
    """Drop an observation's history rows.

    History lives in Postgres regardless of where the observation itself does, and no
    longer cascades from memory_units (that FK was dropped so it could be recorded for
    observations kept outside SQL). Dropped explicitly so a deleted observation's
    snapshots don't accumulate forever.
    """
    await conn.execute(
        f"DELETE FROM {fq_table('observation_history')} WHERE bank_id = $1 AND observation_id = $2",
        bank_id,
        uuid.UUID(observation_id),
    )


async def _execute_delete_action(
    conn: "Connection",
    bank_id: str,
    observation_id: str,
    txn=None,
    expected_revision: str | None = None,
) -> None:
    """Delete a superseded or contradicted observation.

    Blocker 5 (judge 5f7f900d): the target must be deleted through the store CAS seam, never
    by an unconditional row delete. ``cas_delete_memory`` is gated on the Phase-A revision
    token; a concurrently-mutated/deleted target returns STALE/MISSING -> raise
    ``_BatchStaleError`` so the whole original batch rolls back with zero writes (Ruling 1).
    """
    store = get_memories()
    if store.writes_memory_rows_in_sql_for(bank_id):
        if not expected_revision:
            raise _BatchStaleError(f"delete_target_missing_phase_a_revision:{observation_id}")
        outcome = await store.cas_delete_memory(
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            unit_id=observation_id,
            expected_revision=expected_revision,
        )
        if outcome == CASOutcome.STALE:
            logger.warning(
                f"Delete aborted: observation {observation_id} mutated concurrently "
                "(CAS STALE); aborting whole batch with zero writes"
            )
            raise _BatchStaleError(f"delete_target_stale:{observation_id}")
        if outcome == CASOutcome.MISSING:
            logger.warning(
                f"Delete aborted: observation {observation_id} deleted concurrently "
                "(CAS MISSING); aborting whole batch with zero writes"
            )
            raise _BatchStaleError(f"delete_target_missing:{observation_id}")
    else:
        await store.delete_facts(bank_id, [observation_id], txn=txn)
    await _delete_observation_history(conn, bank_id, observation_id)
    logger.debug(f"Deleted observation {observation_id}")


async def _find_related_observations(
    memory_engine: "MemoryEngine",
    bank_id: str,
    query: str,
    request_context: "RequestContext",
    tags: list[str] | None = None,
) -> "RecallResult":
    """
    Find observations related to the given query using optimized recall.

    SECURITY: Filters by tags using all_strict matching to prevent cross-tenant/cross-user
    information leakage. Observations are only consolidated within the same tag scope.

    Uses max_tokens to naturally limit observations (no artificial count limit).
    Includes source memories with dates for LLM context.

    Args:
        tags: Optional tags to filter observations (uses all_strict matching for security)

    Returns:
        List of related observations with their tags, source memories, and dates
    """
    # Use recall to find related observations with token budget
    # max_tokens naturally limits how many observations are returned
    from ...tracing import get_tracer, is_tracing_enabled

    config = await memory_engine._config_resolver.resolve_full_config(bank_id, request_context)

    # SECURITY: Use all_strict matching if tags provided to prevent cross-scope consolidation
    tags_match = "all_strict" if tags else "any"

    # Create span for recall operation within consolidation
    tracer = get_tracer()
    if is_tracing_enabled():
        recall_span = tracer.start_span("hindsight.consolidation_recall")
        recall_span.set_attribute("hindsight.bank_id", bank_id)
        recall_span.set_attribute("hindsight.query", query[:100])  # Truncate for brevity
        recall_span.set_attribute("hindsight.fact_type", "observation")
    else:
        recall_span = None

    # Resolve budget: consolidation doesn't need deep recall, default to LOW to reduce memory fan-out
    recall_budget = Budget(config.consolidation_recall_budget)

    try:
        recall_result = await memory_engine.recall_async(
            bank_id=bank_id,
            query=query,
            budget=recall_budget,
            max_tokens=config.consolidation_max_tokens,  # Token budget for observations (configurable)
            fact_type=["observation"],  # Only retrieve observations
            request_context=request_context,
            tags=tags,  # Filter by source memory's tags
            tags_match=tags_match,  # Use strict matching for security
            include_source_facts=True,  # Embed source facts so we avoid a separate DB fetch
            max_source_facts_tokens=config.consolidation_source_facts_max_tokens,
            max_source_facts_tokens_per_observation=config.consolidation_source_facts_max_tokens_per_observation,
            # Round-robin interleave fusion (no cross-encoder): consolidation is looking
            # for an existing near-identical observation to merge into. Both the
            # cross-encoder (semantic #1 -> reranked #37) and RRF (semantic #1 -> outside
            # the 512-token budget) were measured to bury that twin; interleave guarantees
            # each retrieval arm's top hits a slot, so the semantic-#1 twin is always shown
            # to the LLM, which then UPDATEs instead of creating a duplicate.
            reranking="interleave",
            _quiet=True,  # Suppress logging
        )
    finally:
        if recall_span:
            recall_span.end()

    return recall_result


def _build_observations_for_llm(
    observations: "list[MemoryFact]",
    source_facts: "dict[str, MemoryFact]",
) -> list[dict[str, Any]]:
    """Serialize MemoryFact observations into dicts for the consolidation LLM prompt."""
    obs_list = []
    for obs in observations:
        obs_data: dict[str, Any] = {
            "id": obs.id,
            "text": obs.text,
            "proof_count": len(obs.source_fact_ids or []) or 1,
        }
        if obs.occurred_start:
            obs_data["occurred_start"] = obs.occurred_start
        if obs.occurred_end:
            obs_data["occurred_end"] = obs.occurred_end
        if obs.mentioned_at:
            obs_data["mentioned_at"] = obs.mentioned_at
        source_memories = []
        for sid in obs.source_fact_ids or []:
            sf = source_facts.get(sid)
            if sf is None:
                continue
            sf_data: dict[str, Any] = {"text": sf.text}
            if sf.context:
                sf_data["context"] = sf.context
            if sf.occurred_start:
                sf_data["occurred_start"] = sf.occurred_start
            if sf.occurred_end:
                sf_data["occurred_end"] = sf.occurred_end
            if sf.mentioned_at:
                sf_data["mentioned_at"] = sf.mentioned_at
            source_memories.append(sf_data)
        if source_memories:
            obs_data["source_memories"] = source_memories
        obs_list.append(obs_data)
    return obs_list


def _dedupe_updates(updates: list[_UpdateAction], *, batch_label: str) -> list[_UpdateAction]:
    """Collapse `updates` that target the same `observation_id`.

    LLMs occasionally emit several update entries for one observation in a
    single response (one per facet drawn from the same fact). Without
    deduplication the downstream loop would issue separate DB writes for each
    and the last write would silently overwrite the earlier ones. We keep the
    last text (the LLM's most recent attempt) and union all contributing
    `source_fact_ids`, then warn so the misbehavior is visible in logs.
    """
    if len(updates) < 2:
        return list(updates)

    by_id: dict[str, _UpdateAction] = {}
    collisions = 0
    for upd in updates:
        existing = by_id.get(upd.observation_id)
        if existing is None:
            by_id[upd.observation_id] = upd
            continue
        collisions += 1
        merged_ids = list(dict.fromkeys([*existing.source_fact_ids, *upd.source_fact_ids]))
        by_id[upd.observation_id] = _UpdateAction(
            text=upd.text,
            observation_id=upd.observation_id,
            source_fact_ids=merged_ids,
        )

    if collisions:
        logger.warning(
            f"[CONSOLIDATION] {batch_label}: LLM emitted {collisions} duplicate update(s) targeting "
            f"the same observation_id ({len(updates)} updates -> {len(by_id)} after dedup). "
            "Kept the last text and unioned source_fact_ids."
        )

    return list(by_id.values())


async def _consolidate_batch_with_llm(
    llm_config: Any,
    memories: list[dict[str, Any]],
    union_observations: "list[MemoryFact]",
    union_source_facts: "dict[str, MemoryFact]",
    config: Any,
    remaining_observation_slots: int | None = None,
    max_observations_per_scope: int = -1,
) -> _BatchLLMResult:
    """Single LLM call for a batch of facts against a pooled set of observations."""
    if config is None:
        raise ValueError("config is required for _consolidate_batch_with_llm")
    if union_observations:
        obs_list = _build_observations_for_llm(union_observations, union_source_facts)
        observations_text = json.dumps(obs_list, indent=2, ensure_ascii=False)
    else:
        observations_text = "[]"

    def _fact_line(m: dict[str, Any]) -> str:
        text = f"[{m['id']}] {m['text']}"
        temporal_parts = []
        if m.get("occurred_start"):
            temporal_parts.append(f"occurred_start={m['occurred_start']}")
        if m.get("occurred_end"):
            temporal_parts.append(f"occurred_end={m['occurred_end']}")
        if m.get("mentioned_at"):
            temporal_parts.append(f"mentioned_at={m['mentioned_at']}")
        if temporal_parts:
            text += f" ({', '.join(temporal_parts)})"
        return text

    facts_lines = "\n".join(_fact_line(m) for m in memories)

    # Build capacity note for the prompt when observation limit is configured
    observation_capacity_note: str | None = None
    if remaining_observation_slots is not None and max_observations_per_scope >= 0:
        if remaining_observation_slots == 0:
            observation_capacity_note = (
                f"OBSERVATION LIMIT REACHED ({max_observations_per_scope}/{max_observations_per_scope}). "
                "Only UPDATE or DELETE existing observations. Do NOT create new ones — "
                "merge new knowledge into existing observations via UPDATE."
            )
        elif remaining_observation_slots <= len(memories):
            observation_capacity_note = (
                f"This scope has {remaining_observation_slots} observation slot(s) remaining "
                f"(out of {max_observations_per_scope}). Prefer UPDATE over CREATE when possible."
            )

    # Split the prompt: a bank-agnostic system instruction (rules + input format +
    # decision guide + output format) that is byte-identical across batches AND
    # across banks, and a per-batch user message (mission + capacity note + facts +
    # existing observations). The split lets the system prefix be served from a
    # single Gemini context cache shared by every bank — the bank mission, capacity
    # note, and response_schema (all bank/batch-variable) are kept OUT of the
    # cached prefix so one cache serves all and it never busts within a run.
    system_prompt = build_consolidation_system_prompt(
        llm_output_language=getattr(config, "llm_output_language", None),
    )
    user_content = build_consolidation_input(
        facts_text=facts_lines,
        observations_text=observations_text,
        observations_mission=config.observations_mission,
        observation_capacity_note=observation_capacity_note,
    )

    # Opt into context caching of the stable system prefix when the provider
    # supports it (gemini/vertexai with the flag on). response_schema is NOT
    # passed to the fingerprint: it varies per batch (max_creates) but is not
    # part of the cached prefix, so keying on it would needlessly bust the cache.
    cached_prefix_name: str | None = None
    provider_impl = getattr(llm_config, "_provider_impl", None)
    if provider_impl is not None and provider_impl.supports_prompt_caching():
        try:
            cached_prefix_name = await provider_impl.get_or_create_cached_prefix(
                system_instruction=system_prompt,
            )
        except Exception:
            logger.exception("Consolidation cache prefix lookup failed; falling back to uncached call")
            cached_prefix_name = None

    # Use a constrained response model when observation limit is active
    response_model = _build_response_model(
        max_creates=remaining_observation_slots,
        supports_max_items=config.llm_supports_max_items,
    )

    max_attempts = config.consolidation_max_attempts
    inner_max_retries = config.consolidation_llm_max_retries
    last_exc: Exception | None = None
    # Pre-compute a stable identifier set for the batch so failure logs name the
    # exact memories whose consolidation is failing — without this, an opaque
    # "LLM batch call failed" line gives operators no way to find the offending
    # input until adaptive bisection narrows the batch down to a single memory.
    memory_ids = [str(m.get("id")) for m in memories]
    if len(memory_ids) <= 5:
        ids_label = ", ".join(memory_ids)
    else:
        ids_label = f"{', '.join(memory_ids[:3])}, ... +{len(memory_ids) - 3} more"
    batch_label = f"{len(memory_ids)} memories [{ids_label}]"
    for attempt in range(1, max_attempts + 1):
        try:
            call_kwargs: dict[str, Any] = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "response_format": response_model,
                "scope": "consolidation",
                # Resolved per operation (HINDSIGHT_API_LLM_STRICT_SCHEMA_CONSOLIDATION, falling
                # back to the global flag) so an operator can grammar-enforce consolidation's
                # structured output -- which narrows the raw-JSON failure mode behind #2668 --
                # without forcing strict schema on operations whose model can't satisfy it.
                "strict_schema": config.llm_strict_schema_consolidation,
            }
            # Only request an explicit output budget when configured. Left unset by default the key is
            # omitted, so each provider keeps its implicit default (backwards compatible). Operators on
            # providers with a low hidden cap (notably Bedrock imported models, which truncate structured
            # consolidation JSON) set HINDSIGHT_API_CONSOLIDATION_MAX_COMPLETION_TOKENS to fix it.
            if config.consolidation_max_completion_tokens is not None:
                call_kwargs["max_completion_tokens"] = config.consolidation_max_completion_tokens
            if inner_max_retries is not None:
                call_kwargs["max_retries"] = inner_max_retries
            if cached_prefix_name is not None:
                call_kwargs["cached_prefix"] = cached_prefix_name
            response: _ConsolidationBatchResponse = await llm_config.call(**call_kwargs)
            # Defensive truncation: some LLM providers may not enforce JSON schema max_length
            creates = response.creates
            if remaining_observation_slots is not None and remaining_observation_slots >= 0:
                if len(creates) > remaining_observation_slots:
                    logger.info(
                        f"[CONSOLIDATION] Truncating {len(creates)} creates to {remaining_observation_slots} "
                        f"(max_observations_per_scope={max_observations_per_scope})"
                    )
                    creates = creates[:remaining_observation_slots]
            updates = _dedupe_updates(response.updates, batch_label=batch_label)
            return _BatchLLMResult(
                creates=creates,
                updates=updates,
                deletes=response.deletes,
                obs_count=len(union_observations),
                prompt_chars=len(system_prompt) + len(user_content),
            )
        except Exception as exc:
            last_exc = exc
            logger.warning(
                f"[CONSOLIDATION] LLM batch call failed (attempt {attempt}/{max_attempts}) for {batch_label}: {exc}"
            )

    logger.error(
        f"[CONSOLIDATION] LLM batch call failed after {max_attempts} attempts for {batch_label}, "
        f"skipping batch. Last error: {last_exc}"
    )
    return _BatchLLMResult(
        obs_count=len(union_observations), prompt_chars=len(system_prompt) + len(user_content), failed=True
    )


async def _create_observation_directly(
    pool: DatabaseBackend,
    memory_engine: "MemoryEngine",
    bank_id: str,
    source_memory_ids: list[uuid.UUID],
    observation_text: str,
    tags: list[str] | None = None,
    event_date: datetime | None = None,
    occurred_start: datetime | None = None,
    occurred_end: datetime | None = None,
    mentioned_at: datetime | None = None,
    perf: ConsolidationPerfLog | None = None,
    txn=None,
    conn=None,
    precomputed_embedding: str | None = None,
) -> dict[str, Any]:
    """Create an observation from one or more source memories with pre-processed text.

    The embedding is computed off-connection (a slow embedder must never pin a pooled
    connection); the liveness check + INSERT + observation_sources insert then run in one
    short transaction so they commit atomically.

    ``conn``: when provided (Phase-B caller-owned mode) all writes run on it inside the
    caller's write-group transaction; no transaction is opened here and no connection is
    acquired. When omitted (serial path) a short-lived connection + transaction are opened
    exactly as before.

    ``precomputed_embedding``: Phase-B caller-owned mode passes the embedding already
    computed in Phase A so the embedder never runs under the bank lock (design §4.2).
    """
    # Preflight (non-locking, separate short-lived conn): if every source memory is already
    # gone, skip BEFORE the slow embed — restores the pre-refactor short-circuit so a no-op
    # create doesn't embed and a failing embedder doesn't raise where it used to skip.
    # Skipped in caller-owned mode: Phase B re-validates source liveness fresh under the bank
    # guard inside ``_write_group``, so a second preflight connection would only add lock traffic.
    if conn is None:
        async with acquire_with_retry(pool) as c:
            if not await _any_live_source_memory(c, bank_id, source_memory_ids):
                logger.debug(
                    f"Create skipped: all {len(source_memory_ids)} source memories were deleted before embedding"
                )
                return {"action": "skipped", "reason": "sources_deleted"}

    # Generate embedding for the observation (convert to string for pgvector) BEFORE
    # acquiring a connection so the embedder never holds a pooled connection. In Phase-B
    # caller-owned mode the embedding was already computed in Phase A and is passed in —
    # never run the slow embedder while holding the bank lock.
    t0 = time.time()
    if precomputed_embedding is not None:
        embedding_str = precomputed_embedding
        if perf:
            perf.record_timing("embedding", 0.0)  # measured in Phase A; do not double-count under lock
    else:
        embeddings = await embedding_utils.generate_embeddings_batch(memory_engine.embeddings, [observation_text])
        embedding_str = str(embeddings[0]) if embeddings else None
        if perf:
            perf.record_timing("embedding", time.time() - t0)

    now = datetime.now(timezone.utc)
    obs_event_date = event_date or now
    obs_occurred_start = occurred_start
    obs_occurred_end = occurred_end
    obs_mentioned_at = mentioned_at or now
    obs_tags = tags or []
    observation_id = uuid.uuid4()

    # Write the observation. A SQL store keeps it as a `memory_units` row (inline below, with the
    # search_vector the configured backend needs); a store that owns its rows takes it through
    # upsert_observation as a normal Observation-type memory carrying all of its own state.
    store = get_memories()
    async with _write_group(pool, conn) as conn:
        # FOR SHARE liveness + INSERT share one tiny transaction so a concurrent
        # delete cannot orphan the new observation between the check and the insert.
        live_source_memory_ids = await _filter_live_source_memories(conn, bank_id, source_memory_ids)
        if not live_source_memory_ids:
            logger.debug(f"Create skipped: all {len(source_memory_ids)} source memories were deleted concurrently")
            return {"action": "skipped", "reason": "sources_deleted"}
        source_memory_ids = live_source_memory_ids

        t0 = time.time()
        if store.writes_memory_rows_in_sql_for(bank_id):
            # Query varies based on text search backend.
            from ..schema import _is_oracle  # noqa: PLC0415

            config = get_config()
            if config.text_search_extension == "vchord":
                # VectorChord: manually tokenize and insert search_vector
                query = f"""
                    INSERT INTO {fq_table("memory_units")} (
                        id, bank_id, text, fact_type, embedding, proof_count, source_memory_ids,
                        tags, event_date, occurred_start, occurred_end, mentioned_at, search_vector
                    )
                    VALUES ($1, $2, $3, 'observation', $4::vector, 1, $5, $6, $7, $8, $9, $10,
                            tokenize($3, 'llmlingua2')::bm25_catalog.bm25vector)
                    RETURNING id
                """
            elif config.text_search_extension == "native" and not _is_oracle():
                # Native (PostgreSQL): search_vector is populated with to_tsvector()
                # using the configured native language dictionary, matching the batch
                # insert path in ops_postgresql.insert_facts_batch. On Oracle this falls
                # through to the no-search_vector branch below (Oracle maintains its text
                # index separately; to_tsvector/::regconfig is PG-only — see #3021).
                query = f"""
                    INSERT INTO {fq_table("memory_units")} (
                        id, bank_id, text, fact_type, embedding, proof_count, source_memory_ids,
                        tags, event_date, occurred_start, occurred_end, mentioned_at, search_vector
                    )
                    VALUES ($1, $2, $3, 'observation', $4::vector, 1, $5, $6, $7, $8, $9, $10,
                            to_tsvector('{config.text_search_extension_native_language}'::regconfig, COALESCE($3, '')))
                    RETURNING id
                """
            else:  # pg_textsearch, pgroonga, pg_search, and Oracle: base text columns / separate index
                query = f"""
                    INSERT INTO {fq_table("memory_units")} (
                        id, bank_id, text, fact_type, embedding, proof_count, source_memory_ids,
                        tags, event_date, occurred_start, occurred_end, mentioned_at
                    )
                    VALUES ($1, $2, $3, 'observation', $4::vector, 1, $5, $6, $7, $8, $9, $10)
                    RETURNING id
                """

            row = await conn.fetchrow(
                query,
                observation_id,
                bank_id,
                observation_text,
                embedding_str,
                source_memory_ids,
                obs_tags,
                obs_event_date,
                obs_occurred_start,
                obs_occurred_end,
                obs_mentioned_at,
            )
            created_id = row["id"]

            # Populate observation_sources junction table (Oracle only — PG uses native array ops).
            if memory_engine._backend.ops.uses_observation_sources_table and source_memory_ids:
                await conn.executemany(
                    f"""
                    INSERT INTO {fq_table("observation_sources")} (observation_id, source_id)
                    VALUES ($1, $2)
                    ON CONFLICT (observation_id, source_id) DO NOTHING
                    """,
                    [(observation_id, sid) for sid in dict.fromkeys(source_memory_ids)],
                )
        else:
            await store.upsert_observation(
                conn=conn,
                bank_id=bank_id,
                txn=txn,
                record=FactRecord(
                    unit_id=str(observation_id),
                    text=observation_text,
                    embedding=embedding_str,
                    fact_type="observation",
                    tags=list(obs_tags),
                    proof_count=1,
                    source_memory_ids=[str(s) for s in source_memory_ids],
                    event_date=obs_event_date,
                    occurred_start=obs_occurred_start,
                    occurred_end=obs_occurred_end,
                    mentioned_at=obs_mentioned_at,
                    created_at=now,
                ),
            )
            created_id = observation_id

        if perf:
            perf.record_timing("db_write", time.time() - t0)

    logger.debug(f"Created observation {observation_id} from {len(source_memory_ids)} memories (tags: {obs_tags})")

    return {"action": "created", "observation_id": str(created_id), "tags": obs_tags}
