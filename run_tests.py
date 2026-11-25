"""Run the whole offline test suite: python run_tests.py

Model-based accuracy tests run when the embedding model and the trained head
are available locally and are skipped otherwise. No Ollama server is needed.
"""
import logging
import unittest

logging.disable(logging.CRITICAL)
suite = unittest.defaultTestLoader.discover("tests", top_level_dir=".")
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
