"""Secret scanner — detect and block secrets from entering the vault.

Two-level output (SPEC-10-011 V0.2 / DESIGN-10-011 §3.9):
- `blocking`: real credential findings (private keys, provider/token formats,
  JWT/Bearer, AWS/Google, provable credential-value shapes). Callers treat a
  non-empty blocking list as fail-stop (E-005 in ingest; hard block in
  `vault/lint.py` and `vault/write_ops.py`).
- `warnings`: non-blocking security warnings (sensitive env *names* whose
  values are probe/status fields, booleans/enums, short values, or redacted
  markers). Callers surface these as W-001 without stopping.

`scan_for_secrets` / `is_safe` keep their original blocking-only semantics so
existing callers keep hard-blocking real credentials.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Patterns that indicate secrets
# These are intentionally over-broad — better a false positive than a leak
_SECRET_PATTERNS: list[tuple[str, str]] = [
    # HuggingFace tokens
    (r"hf_[a-zA-Z0-9]{20,}", "HuggingFace token (hf_...)"),
    # OpenAI API keys
    (r"sk-(?:proj-)?[a-zA-Z0-9]{20,}", "OpenAI API key (sk-...)"),
    # Anthropic API keys
    (r"sk-ant-[a-zA-Z0-9]{20,}", "Anthropic API key (sk-ant-...)"),
    # GitHub tokens
    (r"gh[pousr]_[a-zA-Z0-9]{20,}", "GitHub token (ghp_/gho_/ghu_/ghs_/ghr_...)"),
    # Generic bearer tokens
    (r"bearer\s+[a-zA-Z0-9\-_\.]{20,}", "Bearer token"),
    # JWT tokens
    (r"eyJ[a-zA-Z0-9\-_]{20,}\.[a-zA-Z0-9\-_]{20,}\.[a-zA-Z0-9\-_]{20,}", "JWT token"),
    # Private key patterns
    (r"-----BEGIN (?:RSA|EC|DSA|OPENSSH|PGP) PRIVATE KEY-----", "Private key"),
    # AWS keys
    (r"AKIA[0-9A-Z]{16}", "AWS access key"),
    (r"aws_secret_access_key\s*=\s*[\"']?[a-zA-Z0-9+/]{20,}", "AWS secret key"),
    # Generic API key assignments
    (
        r"(?:api[_-]?key|apikey|secret|token|password)\s*[:=]\s*[\"'][a-zA-Z0-9\-_\.]{16,}[\"']",
        "API key/secret assignment",
    ),
    # AgentMail tokens
    (r"am_[a-zA-Z0-9]{20,}", "AgentMail token (am_...)"),
    # Google API keys
    (r"AIza[0-9A-Za-z\-_]{35}", "Google API key"),
]

# Known-good env keys that are never secret regardless of value (DESIGN §3.9.2 rule 1).
_SAFE_ENV_KEYS = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "SHELL",
        "LANG",
        "PWD",
        "TERM",
        "DISPLAY",
        "EDITOR",
        "PAGER",
        "HOSTNAME",
        "LOGNAME",
    }
)

# Status/probe key suffixes (DESIGN §3.9.2 rule 2, SPEC F-112 / appendix C).
# A key ending in one of these defaults to warning unless the value itself is
# credential-shaped.
_STATUS_SUFFIXES = (
    "_PRESENT",
    "_CONFIGURED",
    "_ENABLED",
    "_DISABLED",
    "_ACTIVE",
    "_SET",
    "_REQUIRED",
    "_AVAILABLE",
    "_USED",
)

# Substrings that make an env key name "sensitive" (DESIGN §3.9.2 rule 3).
_SENSITIVE_KEY_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASS", "AUTH", "CREDENTIAL")

# Redacted markers: `***`, `[REDACTED]`, `redacted:...`, `«redacted:...»`.
_REDACTED_RE = re.compile(r"(?:\*\*\*|\[REDACTED\]|redacted[:：]|«[^»]*redacted[^»]*»)", re.IGNORECASE)

# env assignment: KEY=value with optional surrounding quotes. Values are
# captured per-token (quoted text, or a single non-space token) so probe fields
# like `TOKEN_PRESENT=False` classify on the short value, not on everything
# that follows on the line (the old greedy `[^"'\n]{8,}` caused E-005 on real
# yinglong probe messages). Backslash is excluded from the unquoted token
# because Hermes tool outputs JSON-escape newlines as literal `\n`, and those
# must not join several probe fields into one fake long value.
_ENV_ASSIGNMENT = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]{2,})\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s\"'\\\n]*))"
)


@dataclass(frozen=True)
class ScanResult:
    """Two-level scan result.

    blocking: real credential findings -> E-005 fail-stop.
    warnings: non-blocking security warnings -> W-001 + counts (ingest continues).
    """

    blocking: list[str]
    warnings: list[str]


def _mask(value: str) -> str:
    """Mask a matched secret for messages: first4...last4."""
    if len(value) > 8:
        return value[:4] + "..." + value[-4:]
    return "***"


def _looks_redacted(value: str) -> bool:
    return bool(_REDACTED_RE.search(value))


def _value_is_credential_shape(value: str) -> bool:
    """True when an env value itself looks like a real credential (SPEC F-111).

    Heuristics: value length >= 16 AND (matches a known provider/token pattern
    OR has at least two character classes among upper/lower/digit). Short,
    boolean/enum, redacted and low-entropy values are NOT credential-shaped.
    """
    v = value.strip().strip('"').strip("'")
    if not v or len(v) < 16:
        return False
    if any(re.search(pattern, v, re.IGNORECASE) for pattern, _ in _SECRET_PATTERNS):
        return True
    has_upper = any(c.isupper() for c in v)
    has_lower = any(c.islower() for c in v)
    has_digit = any(c.isdigit() for c in v)
    return (has_upper + has_lower + has_digit) >= 2


def classify_env_assignment(key: str, value: str) -> Literal["blocking", "warning", "safe"]:
    """Classify one KEY=value assignment (DESIGN §3.9.2, ordered short-circuit).

    Rules:
    1. Known-good env key -> safe.
    2. Status/probe suffix key -> value-shape check (default warning unless the
       value itself is credential-shaped).
    3. Key without a sensitive word -> safe (the env name alone never produces
       a finding).
    4. Value-shape: credential shape -> blocking; bool/enum/short/redacted/empty
       -> warning.
    """
    key_upper = key.strip().upper()
    if key_upper in _SAFE_ENV_KEYS:
        return "safe"
    if key_upper.endswith(_STATUS_SUFFIXES):
        return "blocking" if _value_is_credential_shape(value) else "warning"
    if not any(word in key_upper for word in _SENSITIVE_KEY_WORDS):
        return "safe"
    return "blocking" if _value_is_credential_shape(value) else "warning"


def _env_spans(content: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _ENV_ASSIGNMENT.finditer(content)]


def _pattern_blocking_findings(content: str, source: str) -> list[str]:
    """Blocking findings from the fixed `_SECRET_PATTERNS` set (never weakened)."""
    findings: list[str] = []
    for pattern, description in _SECRET_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            findings.append(
                f"Secret detected [{description}]: {_mask(match.group(0))} in {source}"
            )
    return findings


def _env_findings(content: str, source: str) -> tuple[list[str], list[str]]:
    """Split env assignments into blocking and warning findings."""
    blocking: list[str] = []
    warnings: list[str] = []
    for match in _ENV_ASSIGNMENT.finditer(content):
        key = match.group(1)
        value = next((g for g in match.groups()[1:] if g is not None), "")
        cls = classify_env_assignment(key, value)
        if cls == "blocking":
            shown = _mask(value) if len(value) > 8 else "***"
            blocking.append(f"Env secret variable: {key}={shown} in {source}")
        elif cls == "warning":
            label = (
                f"Redacted marker for env field {key}"
                if _looks_redacted(value)
                else f"Status/probe env field {key} (non-secret value shape)"
            )
            warnings.append(f"{label} in {source}")
    return blocking, warnings


def _redacted_warnings(content: str, source: str, env_spans: list[tuple[int, int]]) -> list[str]:
    """Standalone redacted markers that are not part of an env assignment."""
    warnings: list[str] = []
    for match in _REDACTED_RE.finditer(content):
        if any(start <= match.start() and match.end() <= end for start, end in env_spans):
            continue
        warnings.append(f"Redacted marker in {source}")
    return warnings


def scan_for_secrets_detailed(content: str, source: str = "<unknown>") -> ScanResult:
    """Full two-level scan.

    Returns ScanResult(blocking, warnings):
    - blocking: real credential findings (E-005 fail-stop for callers).
    - warnings: non-blocking security warnings (W-001, ingest continues).

    Never echoes secret values; messages only carry masked fragments, category
    labels and the source identifier.
    """
    blocking = _pattern_blocking_findings(content, source)
    env_blocking, env_warnings = _env_findings(content, source)
    blocking.extend(env_blocking)
    warnings = list(env_warnings)
    spans = _env_spans(content)
    for warning in _redacted_warnings(content, source, spans):
        if warning not in warnings:
            warnings.append(warning)
    return ScanResult(blocking=blocking, warnings=warnings)


def scan_for_secrets(content: str, source: str = "<unknown>") -> list[str]:
    """Scan content for secrets; returns ONLY blocking findings (backward compat).

    Existing callers (`vault/lint.py`, `vault/write_ops.py`) use this to
    hard-block real credentials. Warning-only content (status/probe fields,
    redacted markers, short/bool/enum values) is not returned here.
    """
    return scan_for_secrets_detailed(content, source).blocking


def scan_file(path: Path) -> list[str]:
    """Scan a file for secrets (blocking-only; same semantics as scan_for_secrets)."""
    if not path.exists():
        return []
    try:
        content = path.read_text(encoding="utf-8")
        return scan_for_secrets(content, str(path))
    except Exception:
        return [f"Could not read {path} for secret scan"]


def is_safe(content: str) -> bool:
    """Quick check: return True if no BLOCKING secrets detected."""
    return len(scan_for_secrets(content)) == 0
