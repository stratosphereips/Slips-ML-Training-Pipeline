# conftest.py - make the repository root importable for the tests
#
# To keep a copy of a test run's output, use e.g.:
#   pytest 2>&1 | tee testing/test_logs/pytest_run.log

import sys
from pathlib import Path


# Project root is one level up from testing/
ROOT = Path(__file__).resolve().parents[1]

# Ensure project root is on sys.path so `import src.<module>` works during collection
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)
