"""Deterministic sensitive-data detection and redaction.

This module runs before a prompt reaches the Jev-Style decision model, Ollama,
or any report file. It uses regular expressions plus a Luhn check only: there is
no model, no network access and no randomness, so the same text always produces
the same redaction.

Detected values are replaced in place with typed placeholders (``<EMAIL_1>``,
``<CREDIT_CARD_1>``, ...). The original value is never returned, logged or
forwarded, and callers only receive the placeholder text plus the set of types
that were found.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

__all__ = ["SENSITIVE_KINDS", "Redaction", "Span", "redact"]


# Order matters: the most specific shapes are replaced first so that, for
# example, a credit card number is never re-detected as a phone number.
SENSITIVE_KINDS = (
    "private_key",
    "jwt",
    "api_key",
    "email",
    "credit_card",
    "aadhaar",
    "ssn",
    "phone",
)

_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_API_KEY_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}"
    r"|ghp_[A-Za-z0-9]{20,}"
    r"|gho_[A-Za-z0-9]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,})\b"
)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_RE = re.compile(
    r"(?<![\d.])(?:\+\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})[ -]\d{3}[ -]\d{4}(?!\.?\d)"
)
# Aadhaar: 12 digits in 4-4-4 form (one consistent separator or none), never starting
# with 0 or 1, and the last digit is a Verhoeff check digit.
_AADHAAR_RE = re.compile(r"(?<![\d.])[2-9]\d{3}([ -]?)\d{4}\1\d{4}(?!\.?\d)")
# Indian mobile: 10 digits starting 6-9, written 98765 43210 or 9876543210, with an
# optional +91 or 0 prefix.
_INDIA_PHONE_RE = re.compile(r"(?<![\d.+])(?:\+91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}(?!\.?\d)")

# Verhoeff tables (dihedral group D5), used by the Aadhaar check digit.
_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6), (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8), (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2), (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4), (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2), (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0), (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5), (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def _luhn(digits: str) -> bool:
    """Return True when ``digits`` satisfies the Luhn checksum."""
    total = 0
    for position, character in enumerate(reversed(digits)):
        try:
            value = int(character)
        except ValueError:
            return False
        if position % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return len(digits) >= 13 and total % 10 == 0


def _valid_card(match: re.Match[str]) -> bool:
    return _luhn(re.sub(r"\D", "", match.group(0)))


def _verhoeff(digits: str) -> bool:
    """Return True when ``digits`` ends in a valid Verhoeff check digit."""
    check = 0
    for position, character in enumerate(reversed(digits)):
        check = _VERHOEFF_D[check][_VERHOEFF_P[position % 8][int(character)]]
    return check == 0


def _valid_aadhaar(match: re.Match[str]) -> bool:
    return _verhoeff(re.sub(r"\D", "", match.group(0)))


_RULES = (
    ("private_key", _PRIVATE_KEY_RE, None),
    ("jwt", _JWT_RE, None),
    ("api_key", _API_KEY_RE, None),
    ("email", _EMAIL_RE, None),
    ("credit_card", _CREDIT_CARD_RE, _valid_card),
    ("aadhaar", _AADHAAR_RE, _valid_aadhaar),
    ("ssn", _SSN_RE, None),
    ("phone", _PHONE_RE, None),
    ("phone", _INDIA_PHONE_RE, None),
)


_DIGITS_ONLY = frozenset({"credit_card", "aadhaar", "ssn", "phone"})


def _digest(kind: str, value: str) -> str:
    """Return a stable digest so a value can be matched without being stored."""
    normalized = re.sub(r"\D", "", value) if kind in _DIGITS_ONLY else value.strip().lower()
    if kind == "phone":
        # The same number with or without a country code (+91, +1) or a leading 0.
        normalized = normalized[-10:]
    return hashlib.sha256(f"{kind}:{normalized}".encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Span:
    """One detected value: its kind and a digest of the value."""

    kind: str
    digest: str


@dataclass(frozen=True)
class Redaction:
    """Result of one redaction pass."""

    pii_types: tuple[str, ...]
    span_count: int
    text: str
    spans: tuple[Span, ...] = ()

    @property
    def detected(self) -> bool:
        return self.span_count > 0


def redact(text: str) -> Redaction:
    """Return ``text`` with every detected sensitive value replaced.

    The returned ``pii_types`` are ordered by first occurrence and contain no
    duplicates; ``span_count`` counts every replaced value.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    if not text.strip():
        return Redaction((), 0, text)

    found: list[str] = []
    spans: list[Span] = []
    counts: dict[str, int] = {}
    total = 0
    redacted_text = text

    for kind, pattern, validator in _RULES:
        def _replace(match: re.Match[str], kind: str = kind, validator=validator) -> str:
            nonlocal total
            if validator is not None and not validator(match):
                return match.group(0)
            total += 1
            spans.append(Span(kind, _digest(kind, match.group(0))))
            index = counts.get(kind, 0) + 1
            counts[kind] = index
            if kind not in found:
                found.append(kind)
            return f"<{kind.upper()}_{index}>"

        redacted_text = pattern.sub(_replace, redacted_text)

    return Redaction(tuple(found), total, redacted_text, tuple(spans))
