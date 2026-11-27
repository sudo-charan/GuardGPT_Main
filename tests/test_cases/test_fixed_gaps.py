"""Gaps that have been fixed: Indian phone and Aadhaar redaction, redacted logs for prompts
that never pass Layer 1, and explicit crisis cases reaching a human reviewer."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core import output_checks
from core import response_composer as composer
from core.sensitive_data import redact
from tests.helpers import ScriptedBackend, StubIntentLayer, make_pipeline, read_jsonl

# Made-up number in Aadhaar format with a valid Verhoeff check digit (not a real person's).
AADHAAR = "2345 6789 0124"


class IndianPhoneTests(unittest.TestCase):
    def test_common_formats_are_redacted(self):
        for value in ("+91 98765 43210", "+91-9876543210", "+919876543210", "09876543210",
                      "9876543210", "98765 43210"):
            result = redact(f"call me on {value}")
            self.assertEqual((result.text, result.pii_types), ("call me on <PHONE_1>", ("phone",)), value)

    def test_ten_digits_not_starting_six_to_nine_are_kept(self):
        self.assertFalse(redact("my order id is 5876543210").detected)

    def test_same_number_matches_with_or_without_country_code(self):
        # The answer check must spot a request value repeated in another format.
        digests = frozenset(span.digest for span in redact("my number is +91 98765 43210").spans)
        checks = output_checks.run("Sure, I will call 9876543210.", request_span_digests=digests)
        self.assertIn("leaked_request_value:phone", checks.pii_leakage.findings)

    def test_us_phone_numbers_still_work(self):
        self.assertEqual(redact("555-123-4567").pii_types, ("phone",))

    def test_number_at_the_end_of_a_sentence_is_redacted(self):
        for value, placeholder in (("555-123-4567", "<PHONE_1>"), ("9876543210", "<PHONE_1>"),
                                   (AADHAAR, "<AADHAAR_1>")):
            self.assertEqual(redact(f"use {value}.").text, f"use {placeholder}.", value)

    def test_decimal_numbers_are_kept(self):
        self.assertFalse(redact("the total was 8765432109.5").detected)


class AadhaarTests(unittest.TestCase):
    def test_valid_number_is_redacted_in_every_layout(self):
        for value in (AADHAAR, AADHAAR.replace(" ", ""), AADHAAR.replace(" ", "-")):
            result = redact(f"my aadhaar is {value}")
            self.assertEqual((result.text, result.pii_types), ("my aadhaar is <AADHAAR_1>", ("aadhaar",)), value)

    def test_wrong_check_digit_is_kept(self):
        self.assertFalse(redact("2345 6789 0125").detected)

    def test_numbers_starting_with_0_or_1_are_kept(self):
        self.assertFalse(redact("1234 5678 9012").detected)

    def test_mixed_separators_are_kept(self):
        self.assertFalse(redact("2345 6789-0124").detected)

    def test_aadhaar_and_phone_together(self):
        result = redact(f"aadhaar {AADHAAR}, phone +91 98765 43210")
        self.assertEqual(result.text, "aadhaar <AADHAAR_1>, phone <PHONE_1>")

    def test_cards_are_still_cards(self):
        self.assertEqual(redact("4111 1111 1111 1111").pii_types, ("credit_card",))


class RedactedLoggingTests(unittest.TestCase):
    def test_over_long_prompt_is_logged_redacted(self):
        with TemporaryDirectory() as folder:
            report = make_pipeline(folder).run("my email is jane@example.com " + "x" * 7000)
            log = (Path(folder) / "guardgpt_prompt.jsonl").read_text(encoding="utf-8")
        self.assertEqual(report["final_status"], "INVALID_INPUT")
        self.assertNotIn("jane@example.com", log)
        self.assertIn("<EMAIL_1>", log)

    def test_prompt_is_logged_redacted_when_layer1_fails(self):
        class BrokenGuard:
            def check(self, *args, **kwargs):
                raise RuntimeError("detector crashed")

        with TemporaryDirectory() as folder:
            app = make_pipeline(folder)
            app.input_guard = BrokenGuard()
            report = app.run("my aadhaar is " + AADHAAR)
            log = (Path(folder) / "guardgpt_prompt.jsonl").read_text(encoding="utf-8")
        self.assertEqual(report["final_status"], "ERROR")
        self.assertNotIn(AADHAAR, log)


class CrisisReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_explicit_crisis_is_queued_as_urgent_and_escalated(self):
        app = make_pipeline(self.root, intent=StubIntentLayer("self_harm_risk", 0.99))
        report = app.run("I want to end my life tonight.", session_id="s")
        review = report["review"]
        self.assertEqual((review["priority"], review["escalated"], review["stage"]), ("urgent", True, "support"))
        self.assertIn("crisis_support", review["escalation_reasons"])
        escalations = read_jsonl(self.root / "guardgpt_escalations.jsonl")
        self.assertEqual(escalations[-1]["review_id"], review["review_id"])

    def test_the_person_still_gets_the_fixed_support_reply_at_once(self):
        backend = ScriptedBackend()
        app = make_pipeline(self.root, intent=StubIntentLayer("self_harm_risk", 0.99), backend=backend)
        report = app.run("I want to end my life tonight.")
        self.assertEqual(report["final_status"], "SUPPORT")
        self.assertEqual(report["response"], composer.CRISIS_REPLY)
        self.assertEqual(report["user_message"], composer.CRISIS_REPLY)
        self.assertEqual(backend.calls, [])        # never model-written

    def test_crisis_review_stores_no_prompt_text(self):
        make_pipeline(self.root, intent=StubIntentLayer("self_harm_risk", 0.99)).run("I want to end my life tonight.")
        for name in ("guardgpt_review.jsonl", "guardgpt_escalations.jsonl"):
            self.assertNotIn("end my life", (self.root / name).read_text(encoding="utf-8"), name)

    def test_support_without_a_crisis_is_not_queued(self):
        backend = ScriptedBackend(["That sounds really hard. Do you want to talk about it?"])
        app = make_pipeline(self.root, intent=StubIntentLayer("self_harm_risk", 0.95), backend=backend)
        report = app.run("Everything feels so heavy lately.")
        self.assertEqual(report["final_status"], "SUPPORT")
        self.assertIsNone(report["review"])


if __name__ == "__main__":
    unittest.main()
