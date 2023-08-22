"""
Processing CLI tool for pytest-profiling data.
Run ``python3 process_prof_data.py -h`` for more information.

This tool takes a folder of pytest-profiling output (a folder of .prof files)
and returns a table of times for each test, sorted highest time to lowest.
If you give a previously generated table under the --compare flag, the tool
adds a column to the table of differences in time taken from that table.
"""
import argparse
from datetime import datetime

import pandas as pd

from profile_utils import CI_profile_summaries, compare_times


def main():
    """Processes, summarises and compares outputs from pytest-profiling."""

    parser = argparse.ArgumentParser(description='Summarise and compare profiling outputs. '
                                                 'Prints output tables and writes csv files.')

    # positional arguments
    parser.add_argument(
        'dir',
        type=str,
        help='the directory of .prof files to be profiled.'
        )

    args = parser.parse_args()
    directory = args.dir

    summary = CI_profile_summaries(directory)

    summary = summary.sort_values(by='tottime', ascending=False)

    print("Profiling results:\n", summary)


if __name__ == "__main__":
    main()
