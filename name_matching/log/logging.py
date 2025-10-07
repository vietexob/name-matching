import logging
import structlog

from tqdm import tqdm
from functools import partialmethod


def configure_structlog(silent: bool=True) -> None:
    """
    Toggle Log Verbosity. Default is NOT Verbose.

    Args:
        silent (bool): sets log level to INFO if True, else sets it to DEBUG
    """

    log_level = logging.INFO if silent else logging.DEBUG
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(log_level))


def configure_tqdm(disable_tqdm: bool=True) -> None:
    """
    Toggle tqdm progress bars. Default is disabled.

    Args:
        disable_tqdm (bool): disables tqdm bars if True, else enables progress bars
    """

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=disable_tqdm)
