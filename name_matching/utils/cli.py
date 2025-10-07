"""
CLI Related Utilities
"""

from argparse import ArgumentParser


def basic_argparser() -> ArgumentParser:
    """
    Configure CLI argument parser with basic parameters
    """

    arg_parser = ArgumentParser()
    arg_parser.add_argument("-s", "--silent", action="store_true")
    arg_parser.add_argument("-hr", "--human-readable", action="store_true")
    arg_parser.add_argument("-dt", "--disable-tqdm", action="store_true")

    return arg_parser
