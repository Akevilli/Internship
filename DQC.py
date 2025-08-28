from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


class BaseOutlierMarker(ABC):
    """
    An abstract base class for outlier detection algorithms.

    This class defines a contract for any class that identifies and marks
    outliers in a pandas DataFrame. Concrete subclasses must implement the
    'mark' method.
    """
    @abstractmethod
    def mark(self, data: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """
        Marks outliers in the provided DataFrame by adding a new boolean column.

        This method should not modify the original DataFrame but should return
        a new DataFrame with the outliers marked.

        Args:
            data (pd.DataFrame): The input DataFrame containing numerical data.

        Returns:
            tuple[pd.DataFrame, str]: A tuple containing:
                - The modified DataFrame with a new boolean column indicating outliers.
                - The name of the new boolean column (e.g., 'is_outlier_IQR').
        """
        pass


class IQROutliersMarker(BaseOutlierMarker):
    def __init__(self, fence: float = 1.5, columns: list = None):
        """
        Initializes the IQR outlier detector.

        Args:
            fence (float): The multiplier for the IQR to set the outlier boundaries.
            columns (list): A list of numerical columns to check for outliers.
        """
        self.fence = fence
        self.columns = columns
        self.outlier_column_name = "is_outlier_IQR"

    def mark(self, data: pd.DataFrame, print_info: bool = False) -> tuple[pd.DataFrame, str]:
        """
        Marks rows that contain outliers based on the IQR method.

        Returns:
            A tuple containing the DataFrame with a new boolean column
            and the name of that new column.
        """
        df = data.copy()

        if not self.columns:
            columns_to_check = df.select_dtypes(include=np.number).columns.tolist()
        else:
            columns_to_check = self.columns

        outlier_mask = pd.Series([False] * len(df), index=df.index)

        for column in columns_to_check:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            column_outlier_mask = (df[column] < Q1 - self.fence * IQR) | (df[column] > Q3 + self.fence * IQR)
            outlier_mask = outlier_mask | column_outlier_mask

        df[self.outlier_column_name] = outlier_mask

        if print_info:
            outlier_count = outlier_mask.sum()
            outlier_percentage = outlier_count / outlier_mask.shape[0]
            print(f"IQR Outliers count: {outlier_count} / {outlier_percentage:.2%}")

        return df, self.outlier_column_name


class ThreeSigmaOutliersMarker(BaseOutlierMarker):
    def __init__(self, columns: list = None):
        """
        Initializes the 3-sigma outlier detector.

        Args:
            columns (list): A list of numerical columns to check for outliers.
        """
        self.columns = columns
        self.outlier_column_name = "is_outlier_3sigma"

    def mark(self, data: pd.DataFrame, print_info: bool = False) -> tuple[pd.DataFrame, str]:
        """
        Marks rows that contain outliers based on the 3-sigma method.

        Returns:
            A tuple containing the DataFrame with a new boolean column
            and the name of that new column.
        """
        df = data.copy()
        outlier_mask = pd.Series([False] * len(df), index=df.index)

        if not self.columns:
            columns_to_check = df.select_dtypes(include=np.number).columns.tolist()
        else:
            columns_to_check = self.columns

        for column in columns_to_check:
            # Note: sigma is standard deviation, not variance.
            std = df[column].std()
            mean = df[column].mean()

            column_outlier_mask = (df[column] < mean - 3 * std) | (df[column] > mean + 3 * std)
            outlier_mask = outlier_mask | column_outlier_mask

        df[self.outlier_column_name] = outlier_mask

        if print_info:
            outlier_count = outlier_mask.sum()
            outlier_percentage = outlier_count / outlier_mask.shape[0]
            print(f"3 sigma Outliers count: {outlier_count} / {outlier_percentage:.2%}")

        return df, self.outlier_column_name


def get_missing_value_info(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and returns a DataFrame with missing value counts and percentages per column.

    Args:
        data (pd.DataFrame): The DataFrame to analyze.

    Returns:
        pd.DataFrame: A summary DataFrame with 'missing_count' and 'missing_percentage' columns.
    """
    missing_info = data.isnull().sum().to_frame('missing_count')
    missing_info['missing_percentage'] = (missing_info['missing_count'] / len(data)) * 100

    print("Missing values per column:")
    print(missing_info)

    return missing_info


def get_duplicate_info(data: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Calculates and returns a DataFrame with information about duplicate rows.

    Args:
        data (pd.DataFrame): The DataFrame to analyze.
        columns (list, optional): A list of columns to consider when checking for duplicates.
            If None, all columns are used.

    Returns:
        pd.DataFrame: A DataFrame containing the duplicate rows.
    """
    duplicates = data.duplicated(subset=columns, keep=False)
    duplicate_rows = data[duplicates]

    total_duplicates = duplicate_rows.shape[0]
    total_percentage = (total_duplicates / data.shape[0]) * 100

    print(f"Found {total_duplicates} duplicate rows ({total_percentage:.2f}%)")

    return duplicate_rows
