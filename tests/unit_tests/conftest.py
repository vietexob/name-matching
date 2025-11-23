"""Unit test base"""
import pytest

from name_matching.config import read_config


@pytest.fixture(scope="module")
def config_ini():
    """Read config files"""

    return read_config()
