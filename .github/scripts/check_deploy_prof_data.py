"""
Tool to combine profiling data into a total summary, and
fail the workflow if it is significantly slower than master.
"""

import argparse
from contextlib import suppress
from datetime import datetime
import sys

import pandas as pd

from profile_utils import percentage_tottime, compare_times


def main():
    """
    Combines profiling data into a total summary, and
    fails the workflow if it is significantly slower than master.
    """
    parser = argparse.ArgumentParser(description='Combine profiling data into a total summary. '
                                                 'Returns an exit code of 1 if significantly '
                                                 'slower than master.')

    parser.add_argument(
        '--files', '-f',
        type=str,
        nargs='+',
        help='The profiling .csv files to combine.'
        )

    parser.add_argument(
        '--name', '-n',
        type=str,
        default=f"profiling-{datetime.now()}",
        help='The name for the output file. Defaults to profiling-[DATE AND TIME]'
        )

    parser.add_argument(
        '--compare', '-c',
        type=str,
        help='if invoked, the results are compared to the previously summarised csv '
             'given under this flag.'
        )

    args = parser.parse_args()
    filename = args.name
    dataframes = [pd.read_csv(file, index_col=0) for file in args.files]

    summary = pd.concat(dataframes, axis=0)

    # recalculate percentage time, compare and sort
    percentage_time = percentage_tottime(summary)
    print(percentage_time)
    summary['% time'] = percentage_time
    if args.compare:
        master = pd.read_csv(args.compare)
        summary = compare_times(master, summary).sort_values(by='change')

    summary = summary.sort_values(by='tottime', ascending=False)

    # get all significantly slower tests
    # if we haven't got a change column (if getting master has failed)
    # pass an empty dataframe and pass 
    try:
        slower = summary[summary['change'] > summary['tottime'] * 0.05]
    except KeyError:
        slower = pd.DataFrame()

    if slower.empty:
        print("Profiling results:\n", summary)
        with open(f'{filename}.csv', 'w', encoding='utf-8') as file:
            # drop change column so this can be used as master summary
            # when branch is deployed
            with suppress(KeyError):
                summary = summary.drop(columns=['change'])
            file.write(summary.to_csv())
        sys.exit(0)

    print("Failing as some tests are over 5% slower than master.\n"
          "These slower tests were the following:\n", slower)
    sys.exit(1)


if __name__ == "__main__":
    main()
