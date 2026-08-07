"""Promotion candidate generation from raw session captures and memory pages.

Scans r0 ingest-sessions output in `raw/sessions/` for repeated facts that may
deserve promotion. The JMap contract is "raw is the immutable source capture";
`brain ingest-sessions` writes new session summaries to `raw/sessions/`
(hermes_sessions._create_session_page).

The legacy `sessions/summaries/` glob is retained as a *read-only* fallback so
older vaults (and any pre-2026-08-05 artifacts) continue to feed promotion
without duplication. New writes must NOT land in `sessions/summaries/`.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from r0b0tlabbra1n.memory.extract import extract_facts

# Preferred scan path (post-2026-08-05 JMap restructure)
_PRIMARY_GLOB = "raw/sessions/*.md"
# Legacy fallback for older vaults that still keep sessions elsewhere
_LEGACY_GLOBS = ("sessions/**/*.md", "sessions/summaries/*.md")


def _iter_session_files(vault_path: Path) -> list[Path]:
    """Return all session markdown files under the vault.

    Prefers the current `raw/sessions/*.md` location; falls back to legacy
    `sessions/` locations so older vaults continue to work.
    """
    seen: set[Path] = set()
    out: list[Path] = []
    primary = sorted(vault_path.glob(_PRIMARY_GLOB))
    if primary:
        for p in primary:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out
    for pattern in _LEGACY_GLOBS:
        for p in sorted(vault_path.glob(pattern)):
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def promotion_candidates(vault_path: Path, min_occurrences: int = 2) -> list[dict]:
    counter: Counter[str] = Counter()
    sources: dict[str, list[str]] = {}
    session_files = _iter_session_files(vault_path)
    if not session_files:
        return []
    for md in session_files:
        try:
            text = md.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for item in extract_facts(text, source=str(md.relative_to(vault_path))):
            key = item.content.strip()
            counter[key] += 1
            sources.setdefault(key, []).append(str(md.relative_to(vault_path)))
    return [
        {"content": k, "occurrences": v, "sources": sources[k]}
        for k, v in counter.most_common()
        if v >= min_occurrences
    ]
