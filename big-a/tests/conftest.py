import pytest
import sys
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _has_qlib_data() -> bool:
    """Return True if Qlib data calendars/day.txt exists in the default data path."""
    day_txt = Path(__file__).resolve().parents[1] / "data" / "qlib_data" / "cn_data" / "calendars" / "day.txt"
    return day_txt.exists()


def init_qlib():
    """Initialize Qlib if data is available. If not, skip the tests that require data."""
    try:
        import qlib  # type: ignore
    except Exception as e:
        pytest.skip(f"QLib is not installed or cannot be imported: {e}")
    if not _has_qlib_data():
        pytest.skip("QLib data not available at ~/.qlib/qlib_data; skipping tests requiring data")
    try:
        qlib.init()
    except Exception as e:
        pytest.skip(f"QLib initialization failed: {e}")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "skip_if_no_data: skip test if Qlib data is not available"
    )


def pytest_runtest_setup(item):
    marker = item.get_closest_marker("skip_if_no_data")
    if marker is not None:
        if not _has_qlib_data():
            pytest.skip("QLib data not available; skipping test with @pytest.mark.skip_if_no_data")


@pytest.fixture(scope="session")
def qlib_initialized():
    """Fixture to ensure Qlib is initialized if data exists."""
    if not _has_qlib_data():
        pytest.skip("QLib data not available; skipping tests requiring Qlib")
    init_qlib()
    yield
