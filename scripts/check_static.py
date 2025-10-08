"""
Deprecated: Static asset smoke-check moved to tests/ and FastAPI app health checks.

Try:
  - pytest -q tests\
  - curl http://localhost:8000/health
  - curl http://localhost:8000/static/<file>
"""

import sys

if __name__ == "__main__":
    print("[deprecated] Static checks are covered by tests and /health. See README for details.")
    sys.exit(0)
