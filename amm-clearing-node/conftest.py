# Makes `src` importable when running pytest from this service directory.
import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"
