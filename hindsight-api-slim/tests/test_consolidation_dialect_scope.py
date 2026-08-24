"""Dialect-scope tests for PostgreSQL-first Store-CAS consolidation v1."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from hindsight_api.engine.consolidation import consolidator as C


@pytest.mark.asyncio
async def test_oracle_rejected_before_consolidation_dependencies():
    resolver = AsyncMock(side_effect=AssertionError("config resolver must not run"))
    llm_wrapper = Mock(side_effect=AssertionError("LLM wrapper must not run"))
    backend = Mock()
    memory_engine = SimpleNamespace(
        _config_resolver=SimpleNamespace(resolve_full_config=resolver),
        _consolidation_llm_config=SimpleNamespace(with_config=llm_wrapper),
        _backend=backend,
    )

    with (
        patch.object(C, "get_config", return_value=SimpleNamespace(database_backend="oracle")),
        patch.object(C, "_run_consolidation_job", new=AsyncMock()) as core,
        pytest.raises(C.UnsupportedConsolidationDialectError, match="PostgreSQL-only"),
    ):
        await C.run_consolidation_job(
            memory_engine=memory_engine,
            bank_id="oracle-bank",
            request_context=object(),
        )

    resolver.assert_not_called()
    resolver.assert_not_awaited()
    llm_wrapper.assert_not_called()
    core.assert_not_called()
    core.assert_not_awaited()
    assert not backend.mock_calls


@pytest.mark.asyncio
async def test_postgresql_still_delegates_to_consolidation_flow():
    resolved_config = SimpleNamespace()
    configured_llm = object()
    resolver = AsyncMock(return_value=resolved_config)
    llm_wrapper = Mock(return_value=configured_llm)
    memory_engine = SimpleNamespace(
        _config_resolver=SimpleNamespace(resolve_full_config=resolver),
        _consolidation_llm_config=SimpleNamespace(with_config=llm_wrapper),
    )
    expected = {"status": "no_new_memories", "bank_id": "pg-bank"}

    with (
        patch.object(C, "get_config", return_value=SimpleNamespace(database_backend="postgresql")),
        patch.object(C, "trace_context_of", return_value=None),
        patch.object(C, "_run_consolidation_job", new=AsyncMock(return_value=expected)) as core,
    ):
        result = await C.run_consolidation_job(
            memory_engine=memory_engine,
            bank_id="pg-bank",
            request_context="ctx",
        )

    assert result == expected
    resolver.assert_awaited_once_with("pg-bank", "ctx")
    llm_wrapper.assert_called_once_with(
        resolved_config,
        bank_id="pg-bank",
        operation="consolidation",
    )
    core.assert_awaited_once()
