"""
Deprecated: This ad-hoc API checker has been superseded by pytest tests in the tests/ folder.

Use one of these instead:
  - pytest -q tests\
  - python -m scripts.dev.run_api_smoke

This script now exits immediately to avoid confusion.
"""

import sys

if __name__ == "__main__":
    print("[deprecated] Use `pytest -q tests/` or `python -m scripts.dev.run_api_smoke` instead.")
    sys.exit(0)
