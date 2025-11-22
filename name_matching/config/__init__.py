from configparser import ConfigParser
from os.path import exists


def read_config(read_env: bool = False) -> ConfigParser:
    """
    Read configuration from ini files based on environment if bool=True

    :param read_env: whether to read environment files, defaults to False
    :type read_env: bool, optional
    :return: config
    :rtype: ConfigParser
    """

    config = ConfigParser()
    config.read("name_matching/config/Config.ini")

    if read_env:
        env_ini = f"name_matching/config/Config.ini"
        if exists(env_ini):
            config.read(env_ini)

    return config
