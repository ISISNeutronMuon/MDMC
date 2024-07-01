"""Functions used by the profile.py CLI tool to process and visualise pytest-profiling data"""

import pstats
import io
import glob

import pandas as pd
from numpy import nan


def CI_profile_summaries(path: str) -> pd.DataFrame:
    """
    Takes a folder of profiling outputs (as given by pytest-profiling)
    and creates a summary of each test.

    Parameters
    ----------
    path: str
        The path to the folder containing profiling outputs.

    Returns
    -------
    pd.DataFrame
        A dataframe of summaries for each test.
    """

    # ensure directory is string-manipulation friendly
    if not path.endswith("/"):
        path += "/"

    # the profiling output folder should not have subfolders,
    # so this should only iterate once
    # it's just an easy way to deal with the tuple unpacking
    # then create generator of summaries for each test file
    files = glob.glob(path + "test_*.prof")
    if files is None:  # if path is empty
        raise OSError("The directory specified does not contain any .prof files.")
    profs = (_summarise(_profile_to_dataframe(file), file[:-5].split("/")[-1])
                for file in files)

    # concatenate summary for each test into a dataframe,
    # then transpose it to be the right way around
    summary = pd.concat(profs, axis=1).T

    # add '% time' column
    percentage_time = percentage_tottime(summary)

    return pd.concat([summary, percentage_time], axis=1)


def _profile_to_dataframe(path: str) -> pd.DataFrame:
    """
    Converts a cProfile output .prof file into a Pandas dataframe.

    Parameters
    ----------
    file: str
        The path to the .prof file.

    Returns
    -------
    pd.Dataframe
        A Pandas dataframe containing the profiling data.
    """
    #import the file as a pstats.Stats object
    out_stream = io.StringIO()
    stats = pstats.Stats(path, stream=out_stream)

    # format data and print to out_stream
    stats.sort_stats("tottime")
    stats.print_stats()

    # convert out_stream data into a CSV and use pandas to convert into a dataframe
    result = out_stream.getvalue()
    # chop off header lines
    result = 'ncalls' + result.split('ncalls')[-1]
    lines = [','.join(line.rstrip().split(None, 5)) for line in result.split('\n')]
    return pd.read_csv(io.StringIO('\n'.join(lines)))

def _summarise(data_frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Summarises a profiling dataframe into the format

    name   | tottime
    -------|---------
    [name] | [time]

    Parameters
    ----------
    df: pd.DataFrame
        A Pandas dataframe.
    name: str
        The name of the test

    Returns
    -------
    pd.Series
        A summarised, one-row series.
    """

    total_time = data_frame['tottime'].sum()

    return pd.Series(data={'name': name,
                           'tottime': total_time},
                     index=['name', 'tottime'])

def percentage_tottime(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the percentage of total time taken by each test.

    Parameters
    ----------
    df: pd.DataFrame
        A dataframe containing a 'tottime' column of total time
        taken by each test.

    Returns
    -------
    pd.Series
        A series named '% time' of each row of tottime converted to a percentage.
    """

    # add percentage of total time column
    sum_time = df['tottime'].sum()
    percentage_time = df['tottime'].apply(lambda x: (x / sum_time) * 100).rename("% time")

    return percentage_time

def compare_times(base: pd.DataFrame, head: pd.DataFrame) -> pd.DataFrame:
    """
    Compares two summary dataframes (as produced by CI_profile_summaries())

    Parameters
    ----------
    base: pd.DataFrame
        The base timings to compare to.
    head: pd.DataFrame
        The timings to compare against the base.

    Returns
    -------
    pd.DataFrame
        The `head` dataframe with an additional column
        containing time differences from `base`
    """

    # get intersection of all tests
    # so we aren't comparing tests that aren't in one of the dataframes
    comparison_df = pd.merge(base, head, how='inner', on=['name'])

    time_difference = (comparison_df['tottime_y']
                       - comparison_df['tottime_x']).rename("change")

    comparison_df = pd.concat([comparison_df, time_difference], axis=1)

    # add back in all rows from head that aren't in base, with time_difference of NaN
    non_match_rows = head[~head['name'].isin(base['name'])]
    nans = pd.Series([nan * len(non_match_rows.index)], name="change")
    non_match_rows = pd.concat([non_match_rows, nans], axis=1)
    comparison_df = pd.concat([comparison_df, non_match_rows], axis=0)

    # then reduce comparison_df to just name and time_difference, and merge
    # back into head

    comparison_df = comparison_df.drop(columns=comparison_df.columns.difference(['name', 'change']))
    return pd.merge(head, comparison_df, on=['name'])
