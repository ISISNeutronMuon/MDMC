import pandas as pd

def filter_dataframe(values, dataframe, column_names=None, column_regex=None,
                     wildcard=None):

    """
    Ignores duplicates

    Parameters
    ----------
    values : iterable
    dataframe : DataFrame
    column_regex : str

    Returns
    -------
    DataFrame
        A DataFrame which has been filtered so that each value in values
        must occur in one of the columns of DataFrame that match
        column_regex
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
        raise ValueError('There must be at least as many columns ({0}) as'
                         ' values ({1})'.format(len(column_names), len(values)))

    # Filter all columns of dataframe which match column_regex for the first
    # value in values
    filtered_dataframe = []
    for col_name in column_names:
        filtered_dataframe.append(dataframe[dataframe[col_name]
                                            == values[0]])
    # Concat the list of filtered dataframes (1 for each matching column)
    # into a single dataframe
    filtered_dataframe = pd.concat(filtered_dataframe)
    # If there is more than one value in values, call _filter_df_multi
    # recursively to further filter by the remaining values
    if len(values) > 1:
        filtered_dataframe = filter_dataframe(values[1:], filtered_dataframe,
                                              column_names=column_names)
    return filtered_dataframe.drop_duplicates()


def filter_ordered_dataframe(values, dataframe, column_names=None,
                             column_regex=None, wildcard=None):

    """
    Ignores duplicates

    Parameters
    ----------
    values : iterable
    dataframe : DataFrame
    column_regex : str

    Returns
    -------
    DataFrame
        A DataFrame which has been filtered so that each value in values
        must occur in one of the columns of DataFrame that match
        column_regex
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
        raise ValueError('There must be at least as many columns ({0}) as'
                         ' values ({1})'.format(len(column_names), len(values)))

    # Whether all elements of each row have the same order as values (including
    # wildcard)
    bool_rows = dataframe[column_names].agg(lambda x: all([x[i] in
                                                           [values[i], wildcard]
                                                           for i
                                                           in range(len(x))]),
                                            axis="columns")
    filtered_dataframe = dataframe.loc[bool_rows]

    return filtered_dataframe.drop_duplicates()
