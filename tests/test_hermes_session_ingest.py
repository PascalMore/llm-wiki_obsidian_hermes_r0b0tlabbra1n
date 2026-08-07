"""Tests for Hermes session ingestion (RFC-10-011 / SPEC-10-011 / DESIGN-10-011).

Unit + fixture coverage:
- U-001..U-008  (schema variant detection, epoch->ISO, mapping, ordering,
  filename uniqueness, provider priority, --since ISO, secret hit)
- F-001..F-007  (real-schema full chain, upstream backward compat, fail-stop,
  idempotence, cursor, secret hit, include-transcripts)
- F-008  (status/probe fields -> W-001 warning, ingest completes, exit 0)
- R-001/R-002  (real Hermes state.db smoke, handled in shell tests)
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from r0b0tlabbra1n.ingest.hermes_sessions import (
    IngestError,
    _create_session_page,
    _detect_schema_variant,
    _derive_provider,
    _epoch_to_iso,
    _normalize_session,
    _sessions_query,
    ingest,
)
from r0b0tlabbra1n.vault.initialize import init_vault


# ---------------------------------------------------------------------------
# Fixtures: real Hermes schema (no created_at, started_at REAL, model only)
# ---------------------------------------------------------------------------


def _create_real_schema_db(
    db_path: Path, num_sessions: int = 3, with_usage: bool = True
) -> list[str]:
    """Build a state.db that mirrors real Hermes schema (no created_at).

    Always creates the session_model_usage table (real Hermes has it). If
    with_usage is True, also inserts the default usage row used by other
    tests; set False when the test wants to seed its own usage rows.
    """
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            started_at REAL NOT NULL,
            model TEXT,
            parent_session_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS session_model_usage (
            session_id TEXT,
            model TEXT,
            billing_provider TEXT,
            first_seen REAL,
            last_seen REAL,
            PRIMARY KEY (session_id, model)
        )
        """
    )

    sids: list[str] = []
    base = 1722733200.0  # 2024-08-04 01:00:00 UTC
    for i in range(num_sessions):
        sid = f"20260804_12514{i}_{'a' * (6 + i)}"  # realistic id format
        sids.append(sid)
        conn.execute(
            "INSERT INTO sessions (id, started_at, model, parent_session_id) "
            "VALUES (?, ?, ?, ?)",
            (sid, base + i * 60.0, f"gpt-5.{i}-terra", ""),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (sid, "user", f"hello from session {i}"),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (sid, "assistant", f"answer for session {i}"),
        )
        if with_usage:
            conn.execute(
                "INSERT INTO session_model_usage "
                "(session_id, model, billing_provider, first_seen, last_seen) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    sid,
                    f"gpt-5.{i}-terra",
                    "openai-codex" if i == 0 else f"provider-{i}",
                    base + i * 60.0,
                    base + i * 60.0 + 30.0,
                ),
            )
    conn.commit()
    conn.close()
    return sids


def _create_upstream_schema_db(db_path: Path, num_sessions: int = 3) -> list[str]:
    """Upstream/r0 contract schema with created_at column."""
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            created_at TEXT,
            parent_session_id TEXT,
            model_name TEXT,
            provider_name TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT
        )
        """
    )
    sids: list[str] = []
    for i in range(num_sessions):
        sid = f"test-session-{i:04d}"
        sids.append(sid)
        conn.execute(
            "INSERT INTO sessions (id, created_at, parent_session_id, "
            "model_name, provider_name) VALUES (?, ?, ?, ?, ?)",
            (sid, f"2026-05-{10 + i:02d}T12:00:00", "", "test-model", "test-provider"),
        )
        for j in range(2):
            conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (sid, "user", f"q{j}-s{i}"),
            )
    conn.commit()
    conn.close()
    return sids


def _create_no_time_col_db(db_path: Path) -> None:
    """Variant C: sessions has no created_at/started_at -> E-002 fail-stop."""
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
    conn.execute(
        "CREATE TABLE messages ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "session_id TEXT, role TEXT, content TEXT)"
    )
    conn.execute("INSERT INTO sessions (id) VALUES ('orphan-sid')")
    conn.commit()
    conn.close()


def _add_usage_rows(conn: sqlite3.Connection, sid: str, rows: list[tuple]) -> None:
    """rows: list of (model, billing_provider, first_seen, last_seen)."""
    for model, prov, first, last in rows:
        conn.execute(
            "INSERT INTO session_model_usage "
            "(session_id, model, billing_provider, first_seen, last_seen) "
            "VALUES (?, ?, ?, ?, ?)",
            (sid, model, prov, first, last),
        )


# ---------------------------------------------------------------------------
# Existing upstream regression tests (must remain green)
# ---------------------------------------------------------------------------


def test_ingest_sessions():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = tmp_path / "state.db"
        _create_upstream_schema_db(db_path, num_sessions=3)
        vault = tmp_path / "test-brain"
        init_vault(vault)

        count = ingest(db_path, vault)
        assert count == 3
        summaries = list((vault / "raw" / "sessions").glob("*.md"))
        assert len(summaries) == 3
        manifest = vault / "_meta" / "ingestion-manifest.jsonl"
        assert manifest.exists()
        lines = manifest.read_text().strip().split("\n")
        assert len(lines) == 3

        count2 = ingest(db_path, vault)
        assert count2 == 0  # idempotent


def test_ingest_sessions_since_cursor():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = tmp_path / "state.db"
        _create_upstream_schema_db(db_path, num_sessions=5)
        vault = tmp_path / "test-brain"
        init_vault(vault)

        count = ingest(db_path, vault, since_cursor="2026-05-13T00:00:00")
        assert count >= 1


# ---------------------------------------------------------------------------
# U-001..U-008 (unit-level contract assertions)
# ---------------------------------------------------------------------------


def test_u_001_schema_variant_detection():
    with tempfile.TemporaryDirectory() as tmp:
        # Variant A: upstream (has created_at)
        a = Path(tmp) / "a.db"
        _create_upstream_schema_db(a)
        conn = sqlite3.connect(str(a))
        assert _detect_schema_variant(conn) == "upstream"
        conn.close()

        # Variant B: hermes (has started_at, no created_at)
        b = Path(tmp) / "b.db"
        _create_real_schema_db(b)
        conn = sqlite3.connect(str(b))
        assert _detect_schema_variant(conn) == "hermes"
        conn.close()

        # Variant C: neither column -> E-002
        c = Path(tmp) / "c.db"
        _create_no_time_col_db(c)
        conn = sqlite3.connect(str(c))
        with pytest.raises(IngestError) as ei:
            _detect_schema_variant(conn)
        assert ei.value.code == "E-002"
        assert ei.value.exit_code == 1
        conn.close()


def test_u_002_epoch_to_iso():
    # +08:00 conversion; microsecond precision; %Y-%m-%d correct
    iso = _epoch_to_iso(1722733200.5)
    assert iso.endswith("+08:00")
    assert iso.startswith("2024-08-04T")
    assert ".500000" in iso
    assert datetime.fromisoformat(iso).strftime("%Y-%m-%d") == "2024-08-04"


def test_u_003_real_schema_column_mapping_no_usage():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        # Skip usage rows entirely.
        _create_real_schema_db(db, num_sessions=1, with_usage=False)
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM sessions LIMIT 1").fetchone()
        norm = _normalize_session(row, "hermes", conn)
        assert norm["id"] == row["id"]
        assert norm["model_name"] == row["model"]
        assert norm["provider_name"] == "unknown"
        # created_at must be ISO str, NOT a float
        assert isinstance(norm["created_at"], str)
        assert "+08:00" in norm["created_at"]
        conn.close()


def test_u_004_deterministic_ordering_same_started_at():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        _create_real_schema_db(db, num_sessions=2)
        # Pin both sessions to identical started_at so order key falls to id
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        ts = 1722733200.0
        conn.execute("UPDATE sessions SET started_at = ?", (ts,))
        conn.commit()
        sql, params = _sessions_query("hermes", None)
        ordered = [r["id"] for r in conn.execute(sql, params).fetchall()]
        # ids ascend regardless of insertion order
        assert ordered == sorted(ordered)
        conn.close()


def test_u_005_filename_uses_full_sid():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        _create_real_schema_db(db, num_sessions=3)
        vault = Path(tmp) / "v"
        init_vault(vault)
        n = ingest(db, vault)
        assert n == 3
        pages = sorted((vault / "raw" / "sessions").glob("*.md"))
        assert len(pages) == 3
        names = [p.name for p in pages]
        # Each filename must embed the FULL sid (not sid[:8]).
        for sid in [f"20260804_12514{i}_{'a' * (6 + i)}" for i in range(3)]:
            matched = [n for n in names if sid in n]
            assert len(matched) == 1, f"sid {sid} should match exactly one page"
        # No truncated sid[:8] collision: all filenames distinct.
        assert len(set(names)) == 3


def test_u_006_provider_priority_from_usage():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        # Skip the default usage row so we control it explicitly here.
        _create_real_schema_db(db, num_sessions=1, with_usage=False)
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        sid = conn.execute("SELECT id FROM sessions LIMIT 1").fetchone()[0]
        # Multiple usage rows: pick the one with the largest COALESCE(last_seen, first_seen, 0)
        _add_usage_rows(
            conn,
            sid,
            [
                ("gpt-5.0-old", "old-provider", 100.0, 110.0),
                ("gpt-5.0-new", "new-provider", 200.0, 300.0),
                ("gpt-5.0-empty", "", 50.0, 400.0),  # empty -> ignored
                ("gpt-5.0-null-last", "null-last", 50.0, None),  # COALESCE -> first_seen
            ],
        )
        conn.commit()
        prov = _derive_provider(conn, sid)
        assert prov == "new-provider"
        conn.close()


def test_u_007_invalid_since_cursor_e007():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        _create_real_schema_db(db, num_sessions=1)
        vault = Path(tmp) / "v"
        init_vault(vault)
        with pytest.raises(IngestError) as ei:
            ingest(db, vault, since_cursor="not-an-iso")
        assert ei.value.code == "E-007"
        assert ei.value.exit_code == 3
        # vault zero writes
        assert list((vault / "raw" / "sessions").glob("*.md")) == []


def test_u_008_secret_hit_e005_no_disk():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        _create_real_schema_db(db, num_sessions=1)
        # Inject a high-entropy token-shaped string that scan_for_secrets will
        # catch (real key-like pattern).
        conn = sqlite3.connect(str(db))
        sid = conn.execute("SELECT id FROM sessions LIMIT 1").fetchone()[0]
        token = "sk-" + "A" * 32 + "bcDEF1234XYZ"
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (sid, "assistant", f"your key is {token}"),
        )
        conn.commit()
        conn.close()
        vault = Path(tmp) / "v"
        init_vault(vault)
        with pytest.raises(IngestError) as ei:
            ingest(db, vault)
        assert ei.value.code == "E-005"
        # Page MUST NOT be on disk
        assert list((vault / "raw" / "sessions").glob("*.md")) == []


# ---------------------------------------------------------------------------
# F-001..F-007 (fixture-level integration)
# ---------------------------------------------------------------------------


def test_f_001_real_schema_full_chain():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        sids = _create_real_schema_db(db, num_sessions=3)
        vault = Path(tmp) / "v"
        init_vault(vault)
        n = ingest(db, vault)
        assert n == 3
        summaries = sorted((vault / "raw" / "sessions").glob("*.md"))
        assert len(summaries) == 3
        manifest_lines = (vault / "_meta" / "ingestion-manifest.jsonl").read_text().strip().splitlines()
        assert len(manifest_lines) == 3
        # frontmatter fields
        text = summaries[0].read_text()
        for required in (
            "title:",
            "created:",
            "type: session",
            "status: active",
            "memory_type: episodic",
            "tier: cold",
            "model:",
            "provider:",
            "session_id:",
            "parent_session_id:",
            "msg_count:",
            "provenance: ingest",
        ):
            assert required in text, f"missing {required}"
        # All full sids present in summary filenames
        names = [p.name for p in summaries]
        for sid in sids:
            assert any(sid in n for n in names), f"missing full sid {sid} in {names}"


def test_f_002_upstream_backward_compat():
    # Already covered by test_ingest_sessions above; this duplicate asserts
    # explicit "upstream" branch on identical fixture shape.
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        _create_upstream_schema_db(db, num_sessions=3)
        vault = Path(tmp) / "v"
        init_vault(vault)
        n = ingest(db, vault)
        assert n == 3


def test_f_003_variant_c_fail_stop_zero_writes():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        _create_no_time_col_db(db)
        vault = Path(tmp) / "v"
        init_vault(vault)
        with pytest.raises(IngestError) as ei:
            ingest(db, vault)
        assert ei.value.code == "E-002"
        assert ei.value.exit_code == 1
        # Vault must have zero writes
        assert list((vault / "raw" / "sessions").glob("*.md")) == []
        manifest = vault / "_meta" / "ingestion-manifest.jsonl"
        if manifest.exists():
            assert manifest.read_text() == ""


def test_f_004_idempotence_after_first_ingest():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        _create_real_schema_db(db, num_sessions=3)
        vault = Path(tmp) / "v"
        init_vault(vault)
        n1 = ingest(db, vault)
        n2 = ingest(db, vault)
        assert n1 == 3
        assert n2 == 0
        # Same number of pages, same manifest size
        assert len(list((vault / "raw" / "sessions").glob("*.md"))) == 3
        manifest_lines = (vault / "_meta" / "ingestion-manifest.jsonl").read_text().strip().splitlines()
        assert len(manifest_lines) == 3


def test_f_005_cursor_only_after_timestamp():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        _create_real_schema_db(db, num_sessions=3)
        vault = Path(tmp) / "v"
        init_vault(vault)
        # Pin sessions to distinct started_at for clean cursor math
        conn = sqlite3.connect(str(db))
        # Second session at +60s, third at +120s (base is 2024-08-04 01:00:00 UTC)
        conn.execute(
            "UPDATE sessions SET started_at = started_at + 60 WHERE rowid = 2"
        )
        conn.execute(
            "UPDATE sessions SET started_at = started_at + 120 WHERE rowid = 3"
        )
        conn.commit()
        conn.close()
        # cursor = base + 30 seconds -> only row 2 and row 3 selected
        cursor = _epoch_to_iso(1722733230.0)
        n = ingest(db, vault, since_cursor=cursor)
        assert n == 2
        # Re-running with same cursor -> idempotent (manifest dedup)
        n2 = ingest(db, vault, since_cursor=cursor)
        assert n2 == 0


def test_f_006_secret_hit_full_ingest():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        _create_real_schema_db(db, num_sessions=2)
        # Inject a token-like string in messages of the first session.
        conn = sqlite3.connect(str(db))
        rows = conn.execute("SELECT id FROM sessions ORDER BY rowid ASC").fetchall()
        target = rows[0][0]
        token = "ghp_" + "B" * 36
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (target, "assistant", f"pastable: {token}"),
        )
        conn.commit()
        conn.close()
        vault = Path(tmp) / "v"
        init_vault(vault)
        with pytest.raises(IngestError) as ei:
            ingest(db, vault)
        assert ei.value.code == "E-005"
        # No partial pages written
        assert list((vault / "raw" / "sessions").glob("*.md")) == []
        # Manifest must not contain the bad session id
        manifest = vault / "_meta" / "ingestion-manifest.jsonl"
        if manifest.exists():
            assert target not in manifest.read_text()


def test_f_007_include_transcripts_writes_raw():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        _create_real_schema_db(db, num_sessions=2)
        vault = Path(tmp) / "v"
        init_vault(vault)
        n = ingest(db, vault, include_transcripts=True)
        assert n == 2
        raw_dir = vault / "raw" / "hermes-sessions"
        assert raw_dir.exists()
        files = list(raw_dir.glob("*.json"))
        assert len(files) == 2
        payload = json.loads(files[0].read_text())
        assert payload["metadata"]["trusted"] is False
        assert "session_id" in payload
        assert isinstance(payload["messages"], list)


def test_f_008_status_field_warning_w001(capsys):
    # SPEC F-008 (V0.2): probe/status fields (TOKEN_PRESENT=False etc.) must
    # NOT trigger E-005. Ingest completes with all pages written, exit 0,
    # W-001 stderr lines, warning count in stdout summary and frontmatter.
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        _create_real_schema_db(db, num_sessions=2)
        # Inject probe/status fields with non-secret values into the first session.
        conn = sqlite3.connect(str(db))
        target = conn.execute("SELECT id FROM sessions ORDER BY rowid ASC").fetchone()[0]
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (target, "tool", "TOKEN_PRESENT=False\nTOKEN_LEN=0\nTUSHARE_VERSION=1.4.24"),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (target, "assistant", "probe result: KEY_CONFIGURED=0"),
        )
        conn.commit()
        conn.close()
        vault = Path(tmp) / "v"
        init_vault(vault)

        n = ingest(db, vault)
        captured = capsys.readouterr()

        # Warning is NOT a failure: all sessions ingested, exit path is normal.
        assert n == 2
        pages = list((vault / "raw" / "sessions").glob("*.md"))
        assert len(pages) == 2
        # stderr carries non-sensitive W-001 lines.
        assert "WARNING W-001" in captured.err
        # stdout summary includes the warning count.
        assert "security warnings" in captured.out
        # frontmatter of the probe session carries a non-sensitive count.
        target_page = next(p for p in pages if target in p.name)
        text = target_page.read_text()
        assert "security_warnings:" in text
        count = int(text.split("security_warnings:")[1].split()[0])
        assert count >= 1
        # Manifest is complete for all sessions.
        manifest_lines = (vault / "_meta" / "ingestion-manifest.jsonl").read_text().strip().splitlines()
        assert len(manifest_lines) == 2


def test_f_009_transcript_warnings_counted_in_stdout(capsys):
    # DESIGN V0.2 §3.9.3: W-001 warnings from BOTH integration points
    # (_create_session_page AND _write_transcript) must accumulate into the
    # stdout summary count. Transcript W-001 lines appear on stderr; the
    # stdout count must include them and the exit path stays normal.
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        _create_real_schema_db(db, num_sessions=2)
        conn = sqlite3.connect(str(db))
        target = conn.execute("SELECT id FROM sessions ORDER BY rowid ASC").fetchone()[0]
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (target, "tool", "TOKEN_PRESENT=False\nTOKEN_LEN=0"),
        )
        conn.commit()
        conn.close()
        vault = Path(tmp) / "v"
        init_vault(vault)

        n = ingest(db, vault, include_transcripts=True)
        captured = capsys.readouterr()

        assert n == 2
        pages = list((vault / "raw" / "sessions").glob("*.md"))
        assert len(pages) == 2
        raw_files = list((vault / "raw" / "hermes-sessions").glob("*.json"))
        assert len(raw_files) == 2
        # Stderr carries W-001 lines for BOTH the page scan and the transcript scan.
        assert captured.err.count("WARNING W-001") >= 2
        # The stdout summary reports a warning count (page + transcript warnings).
        assert "security warnings" in captured.out
        count = int(captured.out.split("security warnings")[0].rsplit("(", 1)[1])
        assert count >= 2
        # Exit path is normal: no IngestError raised, ingest returned the full count.