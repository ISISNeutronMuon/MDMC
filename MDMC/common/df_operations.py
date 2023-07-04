"""Contains some utility functions related to pd.DataFrames, including filtering
functions
"""

from typing import Sequence
import pandas as pd


def filter_dataframe(values: Sequence,
                     dataframe: pd.DataFrame,
                     column_names: 'list[str]' = None,
                     column_regex: str = None) -> pd.DataFrame:
    """
    This filter ignores rows which are duplicated (i.e. it only returns the
    first occurence of any duplicated rows).

    Parameters
    ----------
    values : Sequence
        The values for which to filter. If any of these values occur in any of
        the columns defined by ``column_names`` or ``column_regex``, the row
        will be included in the filtered return.
    dataframe : pandas.DataFrame
        The ``pd.DataFrame`` object to be filtered
    column_names: list, optional
        A `list` of `str` specifying the names of the columns which will be used
        to filter the ``Dataframe``. This cannot be passed if ``column_regex``
        is also passed.
    column_regex : str
        A regular expression matching one or more column names. This specifies
        which columns will be used to filter the ``DataFrame``. This cannot be
        passed if ``column_names`` is also passed.

    Returns
    -------
    pandas.DataFrame
        A ``DataFrame`` which has been filtered so that each value in ``values``
        must occur in one of the columns of ``DataFrame`` that are specified by
        ``column_names`` or matched by ``column_regex``
    """

    if column_names and column_regex:
        raise ValueError('Only one of column_names and column_regex can be'
                         ' passed')
    # Use column names or regex to set column names
    column_names = (column_names if column_names is not None
                    else list(dataframe.filter(regex=column_regex)))

    # Raise an error if there are more values than columns (as every value must
    # be found in a column)
    if len(column_names) < len(values):
        raise ValueError(f'There must be at least as many columns ({len(column_names)}) as'
                         f' values ({len(values)})')

    # Filter all columns of dataframe which match column_regex for the first
    # value in values
    filtered_dataframes: list = []
    for col_name in column_names:
        filtered_dataframes.append(dataframe[dataframe[col_name]
                                            == values[0]])
    # Concat the list of filtered dataframes (1 for each matching column)
    # into a single dataframe
    concat_filtered_dataframe = pd.concat(filtered_dataframes)
    # If there is more than one value in values, call _filter_df_multi
    # recursively to further filter by the remaining values
    if len(values) > 1:
        concat_filtered_dataframe = filter_dataframe(values[1:], concat_filtered_dataframe,
                                              column_names=column_names)
    return concat_filtered_dataframe.drop_duplicates()


def filter_ordered_dataframe(values: Sequence,
                             dataframe: pd.DataFrame,
                             column_names: 'list[str]' = None,
                             column_regex: str = None,
                             wildcard: str = None) -> pd.DataFrame:
    """
    Filters a ``pd.DataFrame`` with an iterable of ordered values. The values
    must occur in columns in the correct order, with the order specified by
    ``column_names``, or by the order which column order which occurs from using
    ``column_regex``.

    This filter ignores rows which are duplicated (i.e. it only returns the
    first occurence of any duplicated rows).

    Parameters
    ----------
    values : Sequence
        The values for which to filter. If any of these values occur in any of
        the columns defined by ``column_names`` or ``column_regex``, the row
        will be included in the filtered return.
    dataframe : pandas.DataFrame
        The ``pd.DataFrame`` object to be filtered
    column_names: list, optional
        A `list` of `str` specifying the names of the columns which will be used
        to filter the ``Dataframe``. This cannot be passed if ``column_regex``
        is also passed.
    column_regex : str
        A regular expression matching one or more column names. This specifies
        which columns will be used to filter the ``DataFrame``. This cannot be
        passed if ``column_names`` is also passed.
    wildcard : str
        A `str` which will be a match in any column

    Returns
    -------
    pandas.DataFrame
        A ``DataFrame`` which has been filtered so that each value in ``values``
        must occur in one of the columns of ``DataFrame`` that are specified by
        ``column_names`` or matched by ``column_regex``
    """

    if column_names and column_regex:
        raise ValueError('Only one of column_names and column_regex can be'
                         ' passed')
    # Use column names or regex to set column names
    column_names = (column_names if column_names is not None
                    else list(dataframe.filter(regex=column_regex)))

    # Raise an error if there are more values than columns (as every value must
    # be found in a column)
    if len(column_names) < len(values):
        raise ValueError(f'There must be at least as many columns ({len(column_names)}) as'
                         f' values ({len(values)})')

    # Whether all elements of each row have the same order as values (including
    # wildcard)
    bool_rows = dataframe[column_names].agg(lambda x: all(x[i] in
                                                          [values[i], wildcard]
                                                          for i
                                                          in range(len(x))),
                                            axis="columns")
    filtered_dataframe = dataframe.loc[bool_rows]

    return filtered_dataframe.drop_duplicates()
