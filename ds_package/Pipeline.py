from typing import Callable

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class GroupAndSummarize(BaseEstimator, TransformerMixin):
    """
    A custom transformer to group data, summarize columns, and add new ones.
    """

    def __init__(self, group_by_col: list[str], agg_dict: dict):
        """
        Initializes the transformer with grouping and aggregation parameters.

        Args:
            group_by_col (list[str]): The column name to group by.
            agg_dict (dict): A dictionary of columns and the aggregation functions to apply.
                             Example: {'sales': 'sum', 'revenue': 'mean'}.
        """
        self.group_by_col = group_by_col
        self.agg_dict = agg_dict

    def fit(self, X: pd.DataFrame, y=None):
        """
        The fit method for transformers. In this case, it doesn't learn anything.

        Args:
            X (pd.DataFrame): The input DataFrame.
            y (pd.Series, optional): The target labels. Not used in this transformer.

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Groups the DataFrame, applies aggregations, and returns the result.

        Args:
            X (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame with grouped and summarized data.
        """
        grouped_df = X.groupby(self.group_by_col).agg(self.agg_dict)
        grouped_df = grouped_df.reset_index()

        for col, func in self.agg_dict.items():
            new_col_name = f"{col}_{func}"
            if new_col_name in grouped_df.columns and 'some_other_col' in grouped_df.columns:
                grouped_df[f"{new_col_name}_ratio"] = grouped_df[new_col_name] / grouped_df['some_other_col']

        return grouped_df


class DateDivider(BaseEstimator, TransformerMixin):
    """
    A transformer that extracts day, month, and year from a date column.

    This class simplifies a time series by converting a single datetime column
    into multiple integer columns, which can be used as features by a machine
    learning model.
    """
    def __init__(self, column: str):
        """
        Initializes the transformer.

        Args:
            column (str): The name of the date column to be transformed.
        """
        self.column = column

    def fit(self, X: pd.DataFrame, y=None):
        """
        The fit method for transformers. In this case, it doesn't learn anything.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts day, month, and year, and drops the original date column.

        Args:
            X (pd.DataFrame): The input DataFrame containing a date column.

        Returns:
            pd.DataFrame: The transformed DataFrame with new date-based features.
        """
        df = X.copy()
        df.dropna(subset=[self.column], inplace=True)

        df["day"] = df[self.column].dt.day.astype("int")
        df["month"] = df[self.column].dt.month.astype("int")
        df["year"] = df[self.column].dt.year.astype("int")

        df = df.drop(columns=[self.column])

        return df


class NanDropper(BaseEstimator, TransformerMixin):
    """
    A transformer that drops rows with missing values.

    This class provides a pipeline-compatible way to handle missing data, either
    by dropping rows with NaNs across all columns or a specified subset.
    """
    def __init__(self, columns: list[str] = None):
        """
        Initializes the transformer.

        Args:
            columns (list[str], optional): A list of columns to check for NaNs.
                If None, rows with NaNs in any column are dropped.
        """
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        """
        The fit method for transformers. In this case, it doesn't learn anything.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drops rows with missing values from the DataFrame.

        Args:
            X (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with rows containing NaNs removed.
        """
        df = X.copy()

        if self.columns is None:
            return df.dropna()

        return df.dropna(subset=self.columns)


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies logarithmic transformation.

    Calculates log function for selected columns, it helps to handle highly right-skewed data
    """
    def __init__(self, columns: list[str], add_one: bool = False):
        """
        Initializes the transformer.

        Args:
            columns (list[str]): A list of columns to check for NaNs.
            add_one (bool, optional): Whether to add one to 0 values. Defaults to False.
        """
        self.columns = columns
        self.add_one = add_one

    def fit(self, X: pd.DataFrame, y=None):
        """
        The fit method for transformers. In this case, it doesn't learn anything.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies logarithmic transformation.

        Args:
            X (pd.DataFrame): The input DataFrame.
        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        df: pd.DataFrame = X.copy()

        for column in self.columns:
            if self.add_one:
                df[f"log_{column}"] = df[column].apply(np.log1p)
            else:
                df[f"log_{column}"] = df[column].apply(np.log)

        return df


class MovingAverageGenerator(BaseEstimator, TransformerMixin):
    """
    A transformer that applies moving average transformation.

    Calculate rolling mean based on previous values.
    """
    def __init__(self, column: str, window_size: int, group_by_col: list[str], sort_by_col: list[str], min_periods: int):
        """
        Initializes the transformer.

        Args:
            column (str): The column for which to generate lag features.
            window_size (int): The size of the moving average window.
            group_by_col (list[str]): The columns to group by before shifting.
            sort_by_col (list[str]): The columns to sort by to ensure correct lag order.
            min_periods (int): The minimum number of observations to consider.
        """
        self.column = column
        self.window_size = window_size
        self.group_by_col = group_by_col
        self.sort_by_col = sort_by_col
        self.min_periods = min_periods

    def fit(self, X: pd.DataFrame, y=None):
        """
        The fit method for transformers. In this case, it doesn't learn anything.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies moving average transformation.

        Args:
            X (pd.DataFrame): The input DataFrame.
        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        df: pd.DataFrame = X.copy()

        df = df.sort_values(self.sort_by_col)

        data = (df
            .groupby(self.group_by_col)[self.column]
            .transform(lambda x: x.rolling(window=self.window_size, min_periods=self.min_periods).mean())
        )

        df[f"MA_{self.column}_{self.window_size}_lags"] = data

        return df


class LagFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    A transformer that generates lag features for a time series.

    This class creates new columns based on past values of a specified column,
    grouped by specified key columns. It's crucial for providing a model with
    historical context in time series forecasting.
    """

    def __init__(self, column: str, lags: list[int], group_by_col: list[str], sort_by_col: list[str]):
        """
        Initializes the transformer.

        Args:
            column (str): The column for which to generate lag features.
            lags (list[int]): A list of integer lags (e.g., [1, 2, 7]) to create.
            group_by_col (list[str]): The columns to group by before shifting.
            sort_by_col (list[str]): The columns to sort by to ensure correct lag order.
        """
        self.column = column
        self.lags = lags
        self.group_by_col = group_by_col
        self.sort_by_col = sort_by_col

    def fit(self, X: pd.DataFrame, y=None):
        """
        The fit method for transformers. In this case, it doesn't learn anything.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generates lag features for the specified column.

        The DataFrame is first sorted, then grouped, and finally the `shift`
        method is used to create lagged versions of the target column.

        Args:
            X (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with new lag feature columns.
        """
        df = X.copy()
        df = df.sort_values(self.sort_by_col)
        for lag in self.lags:
            df[f"{self.column}_lag_{lag}"] = df.groupby(by=self.group_by_col)[self.column].shift(lag)

        return df


class DifferenceGenerator(BaseEstimator, TransformerMixin):
    """
    A transformer that applies N-order difference transformation (Y_t - Y_t-lag)
    for grouped time series data.
    """

    def __init__(
            self,
            columns: list[str],
            lag: int = 1,
            count: int = 1,
            group_by_col: list[str] = None,
            sort_by_col: list[str] = None
    ):
        """
        Initializes the transformer.

        Args:
            columns (list[str]): A list of columns to generate differences.
            lag (int, optional): The step of differencing. Defaults to 1.
            count (int, optional): The number of differences. Defaults to 1.
            group_by_col (list[str], optional): The columns to group by before differencing. Defaults to None.
            sort_by_col (list[str], optional): The columns to sort by to ensure correct order. Defaults to None.
        """
        self.columns = columns
        self.lag = lag
        self.count = count
        self.group_by_col = group_by_col
        self.sort_by_col = sort_by_col

    def fit(self, X: pd.DataFrame, y=None):
        """
        The fit method for transformers. In this case, it doesn't learn anything.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies difference transformation using the transform method for correct alignment.

        Args:
            X (pd.DataFrame): The input DataFrame.
        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        df: pd.DataFrame = X.copy()

        if self.sort_by_col:
            df = df.sort_values(self.sort_by_col)

        for column in self.columns:

            def apply_n_diff(series):
                result = series
                for _ in range(self.count):
                    result = result.diff(self.lag)
                return result

            if self.group_by_col:
                diff_series = df.groupby(self.group_by_col)[column].transform(apply_n_diff)
            else:
                diff_series = apply_n_diff(df[column])

            df[f"{column}_diff_lag{self.lag}_c{self.count}"] = diff_series

        return df


class ConditionDropper(BaseEstimator, TransformerMixin):
    """
    A transformer that drops rows by condition.
    """
    def __init__(self, column: str, condition: Callable):
        """
        Initializes the transformer.

        Args:
            column (str): The column to check for condition.
            condition (Callable): The condition to check.
        """
        self.column = column
        self.condition = condition

    def fit(self, X: pd.DataFrame, y=None):
        """
        The fit method for transformers. In this case, it doesn't learn anything.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drops rows by passed condition.

        Args:
            X (pd.DataFrame): The input DataFrame.
        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        df: pd.DataFrame = X.copy()

        df = df[~self.condition(df[self.column])]
        return df
