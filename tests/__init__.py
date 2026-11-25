# GuardGPT test suite
#
# Tests must never write to the real logs/ folder: that would put test failures
# into the metrics shown by `main.py --status`. Point every log writer at a
# throwaway directory before any test module imports the application.
import atexit
import os
import shutil
import tempfile

# The offline suite targets the rule-based JEV. A configured fine-tuned model
# (JEV_STYLE_MODE=finetuned in .env) would load a 1B-parameter model per process
# and make the suite take about 20 minutes. The environment wins over .env.
os.environ.setdefault("JEV_STYLE_MODE", "builtin")

if "GUARDGPT_LOG_DIR" not in os.environ:
    _LOG_DIR = tempfile.mkdtemp(prefix="guardgpt-test-logs-")
    os.environ["GUARDGPT_LOG_DIR"] = _LOG_DIR
    atexit.register(shutil.rmtree, _LOG_DIR, ignore_errors=True)
