"""Tests for secret scanning.

Covers the existing blocking-only API (backward compat for vault lint /
write_ops) and the two-level classifier added by DESIGN-10-011 §3.9 /
SPEC-10-011 V0.2:
- U-009  status/probe fields -> warning (not blocking)
- U-010  real credentials -> blocking
- U-011  public callers (`scan_for_secrets` / `is_safe`) keep hard-block
- U-012  redacted markers -> warning
- U-013  env value shape classification
"""

import tempfile
from pathlib import Path

from r0b0tlabbra1n.security.secret_scan import (
    classify_env_assignment,
    is_safe,
    scan_file,
    scan_for_secrets,
    scan_for_secrets_detailed,
)


def test_scan_clean_content():
    assert scan_for_secrets("This is clean content.") == []
    assert is_safe("Regular text about machine learning.")


def test_scan_hf_token():
    issues = scan_for_secrets("My HF token is hf_abc123def456ghi789jkl012")
    assert len(issues) >= 1
    assert "hf_..." in issues[0]
    assert not is_safe("hf_abc123def456ghi789jkl012")


def test_scan_openai_key():
    issues = scan_for_secrets("API key: sk-proj-abc123def456ghi789jkl012mno345pqr678")
    assert len(issues) >= 1
    assert "sk-..." in issues[0]


def test_scan_private_key():
    issues = scan_for_secrets(
        "-----BEGIN RSA PRIVATE KEY-----\nsomething\n-----END RSA PRIVATE KEY-----"
    )
    assert len(issues) >= 1
    assert "Private key" in issues[0]


def test_scan_jwt():
    issues = scan_for_secrets(
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    )
    assert len(issues) >= 1
    # Either JWT or Bearer token pattern may match first
    assert any("JWT" in i or "Bearer" in i for i in issues)


def test_scan_env_secret_variable():
    issues = scan_for_secrets('OPENAI_API_KEY="sk-abc123def456ghi789jkl012mno345"')
    assert len(issues) >= 1
    # Either OpenAI key or env variable pattern may match first
    assert any("OPENAI_API_KEY" in i or "OpenAI" in i for i in issues)


def test_scan_env_safe_variables():
    # Safe env vars should not trigger
    content = 'PATH="/usr/bin"\nHOME="/home/user"\nUSER="bob"'
    issues = scan_for_secrets(content)
    assert len(issues) == 0


def test_scan_file_clean():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("This is a clean file.\nNo secrets here.\n")
        f.flush()
        path = Path(f.name)

    try:
        issues = scan_file(path)
        assert issues == []
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# U-009..U-013: two-level classification (DESIGN-10-011 §3.9 / SPEC V0.2)
# ---------------------------------------------------------------------------


def test_u_009_status_probe_fields_warning():
    # Status/probe fields with bool/enum/short values must be warnings, never
    # blocking (old Verify t_d1827953: TOKEN_PRESENT misjudged as E-005).
    for content in (
        "TOKEN_PRESENT=False",
        "TOKEN_PRESENT=true",
        "KEY_CONFIGURED=0",
        "TOKEN_ENABLED=1",
        "FOO_TOKEN=off",
        'API_KEY="***"',
    ):
        result = scan_for_secrets_detailed(content)
        assert result.blocking == [], f"{content!r} must not block"
        assert result.warnings, f"{content!r} should warn"
        # Backward-compat public API: warning-only content never blocks.
        assert scan_for_secrets(content) == [], f"{content!r} blocking-only must be empty"
        assert is_safe(content), f"{content!r} must be safe"


def test_u_010_real_credentials_blocking():
    sk_key = 'OPENAI_API_KEY="sk-' + "A" * 40 + '"'
    hf_token = "hf_" + "B" * 24
    ghp_token = "ghp_" + "C" * 36
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    pem = "-----BEGIN RSA PRIVATE KEY-----\nsomething\n-----END RSA PRIVATE KEY-----"
    aws = "AKIAIOSFODNN7EXAMPLE"
    google = "AIza" + "D" * 35
    high_entropy = "FOO_TOKEN=" + "aB3" * 21 + "xY9"  # 64 chars mixed case+digits
    for content in (sk_key, hf_token, ghp_token, jwt, pem, aws, google, high_entropy):
        result = scan_for_secrets_detailed(content)
        assert result.blocking, f"{content[:20]!r}... must block"
        assert not is_safe(content), f"{content[:20]!r}... must not be safe"


def test_u_011_public_callers_hard_block_unchanged():
    # lint.py / write_ops.py call scan_for_secrets() and treat a non-empty
    # result as a hard block; that must be preserved for real credentials.
    assert scan_for_secrets('OPENAI_API_KEY="sk-' + "A" * 40 + '"') != []
    assert not is_safe("-----BEGIN EC PRIVATE KEY-----")
    # Warning-only content must NOT hard-block those callers.
    assert scan_for_secrets("TOKEN_PRESENT=False") == []
    assert is_safe("TOKEN_PRESENT=False")


def test_u_012_redacted_markers_warning():
    for content in (
        "TOKEN=***",
        'API_KEY="\u00abredacted:...\u00bb"',
        "[REDACTED]",
    ):
        result = scan_for_secrets_detailed(content)
        assert result.blocking == [], f"{content!r} must not block"
        assert result.warnings, f"{content!r} should warn"
        assert is_safe(content)


def test_u_013_env_value_shape_classification():
    # Value shape drives the classification (DESIGN §3.9.2 rule 4).
    assert classify_env_assignment("FOO_TOKEN", "off") == "warning"
    assert classify_env_assignment("FOO_TOKEN", "aB3" * 21 + "xY9") == "blocking"
    assert classify_env_assignment("TOKEN_PRESENT", "False") == "warning"
    assert classify_env_assignment("KEY_CONFIGURED", "0") == "warning"
    # Known-good keys are always safe (rule 1).
    assert classify_env_assignment("PATH", "/usr/bin") == "safe"
    # Keys without sensitive words are safe regardless of value (rule 3).
    assert classify_env_assignment("DATABASE_URL", "postgres://u:p@h/db") == "safe"
    # A real credential value under a sensitive key is blocking.
    assert classify_env_assignment("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY") == "blocking"


def test_u_014_json_escaped_probe_fields_warning():
    # Hermes tool messages JSON-escape newlines as literal `\\n`. The scanner
    # must NOT join several probe fields into one fake long value (this was the
    # real yinglong E-005 trigger for session 20260720_232434_b1bcf1).
    content = (
        '{"output": "TOKEN_PRESENT=False\\nTOKEN_LEN=0\\n'
        'TUSHARE_VERSION=1.4.24\\nAKSHARE_VERSION=1.17.54\\nexit=0", '
        '"exit_code": 0, "error": null}'
    )
    result = scan_for_secrets_detailed(content)
    assert result.blocking == []
    assert result.warnings
    assert is_safe(content)
    assert scan_for_secrets(content) == []
