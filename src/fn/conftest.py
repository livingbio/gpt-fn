from typing import Any

import pytest


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, Any]:
    return {
        "filter_headers": ["authorization"],  # Be sure to match the case of the header exactly
    }
