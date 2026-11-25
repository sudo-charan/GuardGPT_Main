"""Sensitive-data detection, redaction and the no-leak guarantee."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from core.complete_pipeline import AuditLog, CompletePipeline
from core.output_auditor import AuditVerdict
from core.sensitive_data import redact


class RedactionTests(unittest.TestCase):
    def test_redacts_card_ssn_and_email(self):
        result = redact(
            "My card is 4111 1111 1111 1111, SSN 123-45-6789, "
            "email jane.doe@example.com."
        )
        self.assertEqual(result.pii_types, ("email", "credit_card", "ssn"))
        self.assertEqual(result.span_count, 3)
        self.assertNotIn("4111", result.text)
        self.assertNotIn("123-45-6789", result.text)
        self.assertNotIn("jane.doe@example.com", result.text)
        self.assertIn("<CREDIT_CARD_1>", result.text)
        self.assertIn("<SSN_1>", result.text)
        self.assertIn("<EMAIL_1>", result.text)

    def test_clean_prompt_is_unchanged(self):
        text = "Explain how Python lists work."
        result = redact(text)
        self.assertFalse(result.detected)
        self.assertEqual(result.text, text)
        self.assertEqual(result.pii_types, ())

    def test_non_luhn_digit_string_is_left_alone(self):
        result = redact("Order 1234 5678 9012 3456 shipped.")
        self.assertNotIn("credit_card", result.pii_types)
        self.assertEqual(result.text, "Order 1234 5678 9012 3456 shipped.")

    def test_redacts_credentials_and_private_keys(self):
        text = (
            "key sk-abcdefghijklmnopqrstuvwxyz012345 "
            "token eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U "
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----"
        )
        result = redact(text)
        self.assertEqual(
            result.pii_types, ("private_key", "jwt", "api_key")
        )
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz012345", result.text)
        self.assertNotIn("MIIEow", result.text)


if __name__ == "__main__":
    unittest.main()
