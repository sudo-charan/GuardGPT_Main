"""Extended GuardGPT tests: edge cases, multi-turn chats and documented known gaps.

Run them with the rest of the suite (``python run_tests.py``) or on their own:

    python -m unittest discover -s tests/test_cases -t .

Tests marked ``@unittest.expectedFailure`` describe behaviour the project does not have
yet (see ``test_known_gaps.py``). They are reported as "expected failures"; once a gap is
fixed, the run reports an "unexpected success" and the marker should be removed.
"""
