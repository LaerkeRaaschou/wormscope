#!/usr/bin/env bash
# The fast half of the suite — runs before every commit.
#
# tests/unit is Qt-free, and tests/test_contracts.py only checks that the names
# the rest of the code depends on still exist. Together that is a few seconds,
# cheap enough to pay on every commit. The GUI, golden and SAM2 tests wait for
# the pre-push hook.
#
# Coverage is off here (--no-cov) on purpose: a subset of the suite naturally
# covers less than the 85% floor in pytest.ini, and failing a commit over that
# would tell you nothing.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
else
    PY=python3
fi

exec "$PY" -m pytest tests/unit tests/test_contracts.py -q --no-cov
