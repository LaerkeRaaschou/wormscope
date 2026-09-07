#!/usr/bin/env bash
# The whole suite — runs before a push, but only when you are on main.
#
# Side branches are how you share work in progress; blocking those on a green
# suite would only teach you to reach for --no-verify. main is the branch that
# is supposed to always work, so that is the one that is guarded.
#
# Caveat worth knowing: this looks at the branch you are STANDING on, not the
# ref you are pushing to. `git push origin side-branch:main` slips past it.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$BRANCH" != "main" ]; then
    echo "pre-push: on '$BRANCH', not main — skipping the full suite."
    exit 0
fi

if [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
else
    PY=python3
fi

echo "pre-push: on main — running the full suite, coverage floor included…"
exec "$PY" -m pytest
