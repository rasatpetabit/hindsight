"""Task 8b: consolidation_reprepare_attempts is clamped to [0, 8]."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hindsight_api.api.http import BankTemplateConfig
from hindsight_api.config import HindsightConfig


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0", 0),
        ("1", 1),
        ("8", 8),
        ("-1", 0),
        ("99", 8),
    ],
)
def test_from_env_clamps_reprepare_attempts(raw, expected, monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_CONSOLIDATION_REPREPARE_ATTEMPTS", raw)
    config = HindsightConfig.from_env()
    assert config.consolidation_reprepare_attempts == expected


def test_negative_reprepare_does_not_skip_initial_attempt(monkeypatch):
    """``-1`` used to make ``range(1, 1+(-1))`` empty, skipping even the first Phase A."""
    monkeypatch.setenv("HINDSIGHT_API_CONSOLIDATION_REPREPARE_ATTEMPTS", "-1")
    config = HindsightConfig.from_env()
    max_attempts = 1 + int(config.consolidation_reprepare_attempts)
    assert max_attempts >= 1
    assert list(range(1, max_attempts + 1)) == [1]


def test_bank_template_rejects_reprepare_outside_range():
    with pytest.raises(ValidationError):
        BankTemplateConfig(consolidation_reprepare_attempts=-1)
    with pytest.raises(ValidationError):
        BankTemplateConfig(consolidation_reprepare_attempts=9)
    assert BankTemplateConfig(consolidation_reprepare_attempts=0).consolidation_reprepare_attempts == 0
    assert BankTemplateConfig(consolidation_reprepare_attempts=8).consolidation_reprepare_attempts == 8
