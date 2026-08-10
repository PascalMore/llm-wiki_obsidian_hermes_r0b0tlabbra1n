"""Tests for vault initialization."""

import tempfile
from pathlib import Path

import pytest

from r0b0tlabbra1n.paths import VAULT_DIRS
from r0b0tlabbra1n.vault.initialize import init_vault

# V2 raw taxonomy (post-2026-08-10 JMap raw revision). These six roots are
# the only first-level raw roots that `brain init` may create.
_V2_RAW_ROOTS = {
    "raw/documents",
    "raw/datasets",
    "raw/web",
    "raw/media",
    "raw/sessions",
    "raw/skills",
}
# V1 genre roots that are retired and MUST NOT be recreated by init.
_RETIRED_RAW_ROOTS = {
    "raw/articles",
    "raw/papers",
    "raw/projects",
    "raw/assets",
}


def test_init_vault_creates_structure():
    with tempfile.TemporaryDirectory() as tmp:
        vault = Path(tmp) / "test-brain"
        result = init_vault(vault)

        assert result == vault
        assert vault.exists()
        assert (vault / "START_HERE.md").exists()
        assert (vault / "SCHEMA.md").exists()
        assert (vault / "index.md").exists()
        assert (vault / "log.md").exists()
        assert (vault / "_agent" / "START_HERE.md").exists()
        assert (vault / "_agent" / "operating-rules.md").exists()
        assert (vault / "_agent" / "semantic" / "project-status.md").exists()
        assert (vault / "dashboards" / "agent-dashboard.md").exists()
        assert (vault / "_meta" / "vault-state.json").exists()


def test_init_vault_fileexists_error():
    with tempfile.TemporaryDirectory() as tmp:
        vault = Path(tmp) / "test-brain"
        vault.mkdir()
        (vault / "existing.txt").write_text("hello")

        with pytest.raises(FileExistsError):
            init_vault(vault, force=False)


def test_init_vault_force_overwrites():
    with tempfile.TemporaryDirectory() as tmp:
        vault = Path(tmp) / "test-brain"
        vault.mkdir()
        (vault / "existing.txt").write_text("hello")

        result = init_vault(vault, force=True)
        assert result == vault
        # Should have overwritten/created new files
        assert (vault / "START_HERE.md").exists()


def test_vault_dirs_v2_raw_taxonomy_contract():
    """VAULT_DIRS must contain exactly the six V2 raw roots and none of the
    four retired V1 genre roots."""
    raw_entries = {d for d in VAULT_DIRS if d.startswith("raw/")}
    assert _V2_RAW_ROOTS <= raw_entries, (
        f"missing V2 raw roots: {_V2_RAW_ROOTS - raw_entries}"
    )
    leaked = _RETIRED_RAW_ROOTS & raw_entries
    assert not leaked, f"retired V1 raw roots must not appear in VAULT_DIRS: {leaked}"


def test_init_vault_creates_exact_v2_raw_roots():
    """In a fresh temporary vault, all six V2 raw roots must exist and all
    four retired V1 genre roots must be absent."""
    with tempfile.TemporaryDirectory() as tmp:
        vault = Path(tmp) / "v2-brain"
        init_vault(vault)

        for root in _V2_RAW_ROOTS:
            assert (vault / root).is_dir(), f"V2 raw root missing: {root}"

        for root in _RETIRED_RAW_ROOTS:
            assert not (vault / root).exists(), (
                f"retired V1 raw root must not be created: {root}"
            )


def test_init_vault_sessions_and_skills_exist():
    """sessions and skills raw roots must exist (system-special roots)."""
    with tempfile.TemporaryDirectory() as tmp:
        vault = Path(tmp) / "brain"
        init_vault(vault)
        assert (vault / "raw" / "sessions").is_dir()
        assert (vault / "raw" / "skills").is_dir()


def test_init_vault_idempotent():
    """Re-running init with force=True on an already-initialized vault must
    not raise and must leave the V2 roots intact / retired roots absent."""
    with tempfile.TemporaryDirectory() as tmp:
        vault = Path(tmp) / "brain"
        init_vault(vault)
        # Second init with force overwrites templates but must preserve contract.
        init_vault(vault, force=True)
        for root in _V2_RAW_ROOTS:
            assert (vault / root).is_dir()
        for root in _RETIRED_RAW_ROOTS:
            assert not (vault / root).exists()


def test_init_vault_session_capture_path_unchanged():
    """The canonical session capture path `raw/sessions/` must exist after
    init so that ingest can write `raw/sessions/{date}-{sid}.md` without an
    extra mkdir (path preserved across the V2 taxonomy change)."""
    with tempfile.TemporaryDirectory() as tmp:
        vault = Path(tmp) / "brain"
        init_vault(vault)
        capture_dir = vault / "raw" / "sessions"
        assert capture_dir.is_dir()
        # Simulate the ingest write path shape.
        page = capture_dir / "2026-08-10-deadbeef.md"
        page.write_text("ok", encoding="utf-8")
        assert page.exists()


def test_init_vault_no_legacy_sessions_dir():
    """init must NOT create the legacy `sessions/` or `sessions/summaries/`
    directories (only `raw/sessions/` is canonical)."""
    with tempfile.TemporaryDirectory() as tmp:
        vault = Path(tmp) / "brain"
        init_vault(vault)
        assert not (vault / "sessions").exists()
        assert not (vault / "sessions" / "summaries").exists()
