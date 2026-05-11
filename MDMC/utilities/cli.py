# MDMC is a package for the optimisation of classical potentials with experimental data
# Copyright (C) 2026 MDMC Developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Command line interface for MDMC.

Notes
-----
This is only used to provide a simple method for users to run installation tests.
"""
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from textwrap import dedent

from MDMC.utilities.installation_tests import run_installation_tests


def get_parser() -> ArgumentParser:
    """
    Create and return a parser for ``sys.args``.

    Returns
    -------
    ArgumentParser
        The setup argument parser.
    """

    description = "The MDMC command line interface."

    epilog = dedent('''
    Usage Examples
    --------------
    To run all tests of the MDMC installation:

        MDMC test
    ''')

    parser = ArgumentParser(prog='MDMC', description=description, add_help=True,
                            epilog=epilog, formatter_class=RawTextHelpFormatter)
    parser.set_defaults(func=parser.print_help)
    subparsers = parser.add_subparsers()
    _add_test_subparser(subparsers)

    return parser

# subparsers not typed because it is protected class (special action object)


def _add_test_subparser(subparsers: ArgumentParser) -> None:
    """
    Add a subparser for running installation tests.

    Parameters
    ----------
    subparsers : ArgumentParser
        Subparser for installation tests.
    """

    test_help = '''
    This is used to test the installation of the MDMC core
    components and optional functionality. If a test fails, please consult the
    log file for further details.'''

    test_subparser = subparsers.add_parser('test', help=test_help,
                                           formatter_class=RawTextHelpFormatter)
    test_subparser.set_defaults(func=run_installation_tests)


def main():
    """
    Entry point exposed for running installation tests.
    """

    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.func()
