# pylint: disable=missing-module-docstring,missing-function-docstring

from configparser import ConfigParser
from unittest.mock import patch

from name_matching.config import read_config


def test_read_config_default():
    """Test reading config without environment flag"""
    config = read_config(read_env=False)

    assert isinstance(config, ConfigParser)
    assert config.sections()


def test_read_config_with_env_flag_file_exists():
    """Test reading config with environment flag when env file exists"""
    # Create a temporary env config file

    # The file already exists, so we just test that it gets read
    with patch("name_matching.config.exists", return_value=True):
        config = read_config(read_env=True)
        assert isinstance(config, ConfigParser)


def test_read_config_with_env_flag_file_not_exists():
    """Test reading config with environment flag when env file doesn't exist"""
    with patch("name_matching.config.exists", return_value=False):
        config = read_config(read_env=True)
        assert isinstance(config, ConfigParser)
