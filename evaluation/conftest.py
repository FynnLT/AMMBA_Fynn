"""Import paths for the evaluation test suite.

The harness modules import each other flatly (`import runner`, `import stack`)
because they are run as scripts from inside `harness/`. Under pytest the
rootdir is `evaluation/`, so those names are not importable unless the two
package-less directories are on `sys.path`. Adding them here keeps the
scripts runnable both ways and avoids turning the harness into a package,
which would change every existing call site for no gain.

`calibration/` is added when it exists: it arrives with T-19, after the
harness sections, and a missing directory must not break the harness tests.
`analysis/` holds the DR2-DR5 analysis over recorded runs, imported flatly
the same way.
"""
import sys
from pathlib import Path

import pytest

EVALUATION_DIR = Path(__file__).resolve().parent

for name in ("harness", "calibration", "analysis"):
    directory = EVALUATION_DIR / name
    if directory.is_dir():
        path = str(directory)
        if path not in sys.path:
            sys.path.insert(0, path)


@pytest.fixture
def anyio_backend():
    """Same async-test idiom as the three services: `@pytest.mark.anyio`
    plus this fixture, no extra plugin."""
    return "asyncio"
