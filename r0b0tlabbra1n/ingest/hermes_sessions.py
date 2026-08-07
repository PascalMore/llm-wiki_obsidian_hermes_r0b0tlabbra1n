"""Hermes session ingestion — read state.db and convert to vault pages."""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, NoReturn

from r0b0tlabbra1n.security.raw_policy import raw_metadata
from r0b0tlabbra1n.security.secret_scan import scan_for_secrets_detailed


# Asia/Shanghai fixed timezone for Hermes session id and started_at epoch.
# RFC Q-1 closed: hermes session ids embed local +08:00 time, and `started_at`
# epoch is UTC seconds; converting with +08:00 keeps `created_at` ISO strings
# consistent with the date prefix used in summary filenames.
_TZ = timezone(timedelta(hours=8))


class IngestError(SystemExit):
    """Contract error: code in {E-001..E-007}; exit_code is the process exit code.

    Inherits SystemExit so cli.py's existing "uncaught exception -> exit 1" path
    naturally propagates the contract exit_code (E-001..E-006 -> 1, E-007 -> 3)
    without modifying cli.py.
    """

    def __init__(self, code: str, exit_code: int, message: str) -> None:
        super().__init__(exit_code)
        self.code = code
        self.exit_code = exit_code
        self.message = message


def _abort(code: str, exit_code: int, message: str) -> NoReturn:
    """Emit machine-readable error line to stderr then raise IngestError."""
    print(f"ERROR {code}: {message}", file=sys.stderr)
    raise IngestError(code, exit_code, message)


def ingest(
    state_db_path: Path,
    vault_path: Path,
    since_cursor: str | None = None,
    include_transcripts: bool = False,
) -> int:
    """Ingest Hermes sessions from state.db into the brain vault.

    Hermes `sessions` schema is detected at runtime:
    - Variant "upstream": columns include `created_at` (r0 test/fixture contract).
    - Variant "hermes": real schema uses `started_at REAL` + `model`, no
      `provider_name`; provider is derived from `session_model_usage`.
    - Variant C (no usable time column) is rejected with E-002 before any
      vault write.

    The Hermes SQLite file is opened read-only via `file:...?mode=ro`; the
    adapter performs zero writes/schema changes on the source.
    """
    vault_path = Path(vault_path)
    manifest_path = vault_path / "_meta" / "ingestion-manifest.jsonl"
    ingested_ids = _load_ingested(manifest_path)

    uri = f"file:{state_db_path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        _abort("E-006", 1, f"state.db open failed: {exc}")  # raises; conn unbound on this path
    conn.row_factory = sqlite3.Row
    try:
        variant = _detect_schema_variant(conn)

        if since_cursor is not None:
            try:
                datetime.fromisoformat(since_cursor)
            except (ValueError, TypeError) as exc:
                _abort("E-007", 3, f"invalid --since cursor {since_cursor!r}: {exc}")

        sql, params = _sessions_query(variant, since_cursor)
        try:
            sessions = conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            _abort("E-003", 1, f"sessions query failed: {exc}")

        count = 0
        total_warnings = 0
        for row in sessions:
            sid = row["id"]
            if sid in ingested_ids:
                continue
            try:
                norm = _normalize_session(row, variant, conn)
                page, page_warnings = _create_session_page(conn, norm, vault_path)
                transcript_warnings = 0
                if page and include_transcripts:
                    transcript_warnings = _write_transcript(conn, norm, state_db_path, vault_path)
                if page:
                    _record_ingested(manifest_path, sid)
                    count += 1
                total_warnings += page_warnings + transcript_warnings
            except IngestError:
                # Contract errors are fail-stop: re-raise so the process exits
                # non-zero without partial vault pollution beyond what already
                # landed on disk for prior successful sessions.
                raise
            except OSError as exc:
                _abort(
                    "E-004",
                    1,
                    f"vault write failed for session {sid}: {exc}",
                )
        if total_warnings > 0:
            print(f"Ingested {count} sessions ({total_warnings} security warnings).")
    finally:
        conn.close()
    return count


def _detect_schema_variant(conn: sqlite3.Connection) -> str:
    """Detect Hermes `sessions` schema variant: "upstream" / "hermes" / fail-stop."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = {"sessions", "messages"} - tables
    if missing:
        _abort("E-001", 1, f"missing tables: {sorted(missing)}")

    mcols = {r[1] for r in conn.execute("PRAGMA table_info(messages)")}
    if not {"session_id", "role", "content"}.issubset(mcols):
        _abort("E-001", 1, "messages missing required columns session_id/role/content")

    scols = {r[1] for r in conn.execute("PRAGMA table_info(sessions)")}
    if "created_at" in scols:
        return "upstream"
    if "started_at" in scols:
        return "hermes"
    _abort(
        "E-002",
        1,
        "no usable time column (created_at/started_at) on sessions table",
    )


def _sessions_query(
    variant: str, since_cursor: str | None
) -> tuple[str, list[Any]]:
    """Build the variant-specific SELECT statement with deterministic ordering.

    Cursor semantics: the cursor is an ISO-8601 (+08:00) string. For the
    upstream variant (created_at TEXT) we compare lexicographically; for the
    hermes variant (started_at REAL epoch seconds) we convert the cursor to
    epoch seconds and bind as REAL.
    """
    cursor_value: Any = None
    if variant == "upstream":
        sql = (
            "SELECT id, created_at, parent_session_id, model_name, provider_name "
            "FROM sessions"
        )
        time_col = "created_at"
        cursor_value = since_cursor
    else:  # hermes
        sql = (
            "SELECT id, started_at, parent_session_id, model, NULL AS provider_name "
            "FROM sessions"
        )
        time_col = "started_at"
        if since_cursor is not None:
            cursor_dt = datetime.fromisoformat(since_cursor)
            # Compare in epoch seconds (UTC) so the SQL bound param is REAL.
            cursor_value = cursor_dt.timestamp()

    params: list[Any] = []
    if cursor_value is not None:
        sql += f" WHERE {time_col} > ?"
        params.append(cursor_value)
    sql += f" ORDER BY {time_col} ASC, id ASC"
    return sql, params


def _epoch_to_iso(epoch: float) -> str:
    """Convert UTC epoch seconds to +08:00 ISO-8601 with microsecond precision."""
    return datetime.fromtimestamp(float(epoch), tz=_TZ).isoformat()


def _derive_provider(conn: sqlite3.Connection, sid: str) -> str:
    """Pick the most recent non-empty billing_provider for this session.

    Returns "unknown" if session_model_usage is missing or empty.
    """
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    except sqlite3.Error:
        return "unknown"
    if "session_model_usage" not in tables:
        return "unknown"
    try:
        row = conn.execute(
            "SELECT billing_provider FROM session_model_usage "
            "WHERE session_id = ? AND billing_provider != '' "
            "ORDER BY COALESCE(last_seen, first_seen, 0) DESC, model ASC "
            "LIMIT 1",
            (sid,),
        ).fetchone()
    except sqlite3.Error:
        return "unknown"
    if row is None:
        return "unknown"
    val = row["billing_provider"]
    return val if val else "unknown"


def _normalize_session(
    row: sqlite3.Row, variant: str, conn: sqlite3.Connection
) -> dict[str, Any]:
    """Map a raw sessions row into the canonical dict consumed downstream.

    Returns a dict with: id, created_at (ISO str), parent_session_id,
    model_name, provider_name. All fields defaulted to safe values when NULL.
    """
    if variant == "upstream":
        created_at = row["created_at"] or _epoch_to_iso(datetime.now().timestamp())
        model_name = row["model_name"] or "unknown"
        provider_name = row["provider_name"] or "unknown"
    else:  # hermes
        started_at = row["started_at"]
        if started_at is None:
            # No usable time -> E-002 fail-stop (never ingested silently).
            _abort(
                "E-002",
                1,
                f"session {row['id']!r} has NULL started_at",
            )
        created_at = _epoch_to_iso(started_at)
        model_name = row["model"] or "unknown"
        provider_name = _derive_provider(conn, row["id"])

    return {
        "id": row["id"],
        "created_at": created_at,
        "parent_session_id": row["parent_session_id"] or "",
        "model_name": model_name,
        "provider_name": provider_name,
    }


def _load_ingested(manifest_path: Path) -> set[str]:
    ingested = set()
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("session_id"):
                ingested.add(entry["session_id"])
    return ingested


def _record_ingested(manifest_path: Path, session_id: str) -> None:
    entry = json.dumps({"session_id": session_id, "ingested_at": datetime.now().isoformat()})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(entry + "\n")


def _messages(conn: sqlite3.Connection, sid: str) -> list[sqlite3.Row]:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(messages)")}
    if {"session_id", "role", "content"}.issubset(cols):
        return conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC", (sid,)
        ).fetchall()
    return []


def _summarize_messages(messages: list[sqlite3.Row]) -> dict[str, list[str]]:
    summary = {
        "requests": [],
        "actions": [],
        "decisions": [],
        "failures": [],
        "commands": [],
        "followups": [],
    }
    for m in messages:
        role = m["role"]
        content = (m["content"] or "").strip()
        if not content:
            continue
        low = content.lower()
        first = " ".join(content.split())[:180]
        if role == "user":
            summary["requests"].append(first)
        elif "error" in low or "traceback" in low or "failed" in low:
            summary["failures"].append(first)
        elif "decision" in low or "decided" in low:
            summary["decisions"].append(first)
        elif "```" in content or content.startswith("$"):
            summary["commands"].append(first)
        else:
            summary["actions"].append(first)
    return {k: v[:8] for k, v in summary.items()}


def _section(title: str, items: list[str]) -> str:
    if not items:
        return f"## {title}\n\nNone detected.\n"
    lines = "\n".join(f"- {item}" for item in items)
    return f"## {title}\n\n{lines}\n"


def _create_session_page(
    conn: sqlite3.Connection, session: dict[str, Any], vault_path: Path
) -> tuple[Path | None, int]:
    """Write a vault summary page for the normalized session dict.

    Filename uses the full session id (no [:8] truncation) to avoid same-day
    collisions: `{YYYY-MM-DD}-{sid}.md`. Two-level secret scan (DESIGN §3.9):
    blocking findings -> E-005 fail-stop (page NOT written); warning findings
    -> W-001 stderr lines + `security_warnings` frontmatter count, page written
    normally. Returns (page_path, warning_count).
    """
    sid = session["id"]
    created = session["created_at"]
    model = session["model_name"] or "unknown"
    provider = session["provider_name"] or "unknown"
    parent_id = session["parent_session_id"] or ""
    messages = _messages(conn, sid)
    summary = _summarize_messages(messages)
    try:
        date_prefix = datetime.fromisoformat(created).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        date_prefix = "unknown"
    # Canonical JMap raw capture location (post-2026-08-05 governance):
    # session summaries are immutable source captures and MUST be written
    # under `raw/sessions/`. The historical `sessions/summaries/` path is
    # no longer created by ingest; legacy files there remain readable via
    # `promote_candidates._iter_session_files` for backward compatibility.
    page_dir = vault_path / "raw" / "sessions"
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{date_prefix}-{sid}.md"
    parent_ref = f"\n- Parent session: `{parent_id}`" if parent_id else ""
    content = f"""---
title: Session {sid[:12]}
created: {created}
type: session
status: active
memory_type: episodic
tier: cold
model: {model or "unknown"}
provider: {provider or "unknown"}
session_id: "{sid}"
parent_session_id: "{parent_id or ""}"
msg_count: {len(messages)}
provenance: ingest
---

# Session {sid[:12]}

- **Date:** {created}
- **Model:** {model or "unknown"}
- **Provider:** {provider or "unknown"}
- **Messages:** {len(messages)}{parent_ref}

{_section("User Requests", summary["requests"])}
{_section("Actions Taken", summary["actions"])}
{_section("Decisions", summary["decisions"])}
{_section("Failures", summary["failures"])}
{_section("Commands", summary["commands"])}
{_section("Follow-ups", summary["followups"])}
## Provenance

- Source: Hermes state.db read-only ingest
- Session ID: `{sid}`
- Review status: generated summary, needs human review

[[../transcripts-index]]
"""
    result = scan_for_secrets_detailed(content, str(page_path))
    if result.blocking:
        _abort(
            "E-005",
            1,
            f"Secrets detected while ingesting session {sid}: {'; '.join(result.blocking)}",
        )
    warning_count = 0
    for warning in result.warnings:
        print(f"WARNING W-001: {warning} for session {sid}", file=sys.stderr)
        warning_count += 1
    if warning_count:
        content = content.replace(
            "provenance: ingest\n",
            f"provenance: ingest\nsecurity_warnings: {warning_count}\n",
            1,
        )
    page_path.write_text(content, encoding="utf-8")
    return page_path, warning_count


def _write_transcript(
    conn: sqlite3.Connection, session: dict[str, Any], state_db_path: Path, vault_path: Path
) -> int:
    """Write the raw transcript JSON for a session.

    Two-level secret scan (DESIGN §3.9.3): blocking findings -> E-005 fail-stop
    (transcript NOT written); warning findings -> W-001 stderr lines, transcript
    written normally. Returns the W-001 warning count (0 if none) so ingest()
    can accumulate it into the stdout summary.
    """
    sid = session["id"]
    messages = [{"role": m["role"], "content": m["content"]} for m in _messages(conn, sid)]
    payload = {
        "metadata": raw_metadata(str(state_db_path)),
        "session_id": sid,
        "messages": messages,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    result = scan_for_secrets_detailed(text, f"raw transcript {sid}")
    if result.blocking:
        _abort(
            "E-005",
            1,
            f"Secrets detected in raw transcript {sid}: {'; '.join(result.blocking)}",
        )
    warning_count = 0
    for warning in result.warnings:
        print(f"WARNING W-001: {warning} for session {sid}", file=sys.stderr)
        warning_count += 1
    raw_dir = vault_path / "raw" / "hermes-sessions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / f"{sid}.json"
    out.write_text(text, encoding="utf-8")
    return warning_count