import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod
from DQC import BaseOutlierMarker


class BaseExtractor(ABC):
    """
    An abstract base class for data extraction.

    Concrete subclasses must implement the 'extract' method
    to retrieve data from a source.
    """
    @abstractmethod
    def extract(self) -> pd.DataFrame:
        """
        Abstract method to extract data.

        Returns:
            pd.DataFrame: The raw data as a pandas DataFrame.
        """
        pass


class BaseTransformer(ABC):
    """
    An abstract base class for a data transformation pipeline.

    Concrete subclasses must implement 'fit' to learn from data
    and 'transform' to apply transformations.
    """
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to apply a series of transformations to a DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to be transformed.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        pass

    def fit(self, data: pd.DataFrame):
        """
        Base method to fit transformations on a DataFrame.

        This method is responsible for learning parameters from the data
        that will be used in the 'transform' method.
        """
        pass


class BaseLoader(ABC):
    """
    An abstract base class for loading processed data.

    Concrete subclasses must implement the 'load' method
    to write a DataFrame to a destination.
    """
    @abstractmethod
    def load(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to load a DataFrame to a destination.

        Args:
            data (pd.DataFrame): The DataFrame to be loaded.
        """
        pass


class SimpleTransformation(ABC):
    """
    An abstract base class for simple transformations.

    These transformations do not require fitting and only have a 'transform' method.
    """
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a transformation to a DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to be transformed.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        pass


class FittedTransformation(ABC):
    """
    An abstract base class for transformations that require fitting.

    These transformations must have both 'fit' and 'transform' methods.
    """
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a fitted transformation to a DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to be transformed.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        pass

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """
        Fits the transformation on the provided data to learn parameters.

        Args:
            data (pd.DataFrame): The DataFrame used to fit the transformation.
        """
        pass


class Normalizer(FittedTransformation):
    """
    Normalizes specified columns using MinMaxScaler.

    Args:
        columns (list): A list of column names to normalize.
    """
    def __init__(self, columns: list):
        self.columns = columns
        self.normalizers = {column: MinMaxScaler() for column in columns}

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the fitted normalization to the specified columns.

        Args:
            data (pd.DataFrame): The DataFrame to be transformed.

        Returns:
            pd.DataFrame: The DataFrame with normalized columns.
        """
        df = data.copy()

        for column in self.columns:
            df[column] = self.normalizers[column].transform(df[[column]])

        return df

    def fit(self, data: pd.DataFrame):
        """
        Fits the MinMaxScaler on the specified columns.

        Args:
            data (pd.DataFrame): The DataFrame used to fit the scaler.
        """
        for column in self.columns:
            self.normalizers[column].fit(data[[column]])


class Standardizer(FittedTransformation):
    """
    Standardizes specified columns using StandardScaler.

    Args:
        columns (list): A list of column names to standardize.
    """
    def __init__(self, columns: list):
        self.columns = columns
        self.standardizers = {column: StandardScaler() for column in columns}

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the fitted standardization to the specified columns.

        Args:
            data (pd.DataFrame): The DataFrame to be transformed.

        Returns:
            pd.DataFrame: The DataFrame with standardized columns.
        """
        df = data.copy()

        for column in self.columns:
            df[column] = self.standardizers[column].transform(df[[column]])

        return df

    def fit(self, data: pd.DataFrame):
        """
        Fits the StandardScaler on the specified columns.

        Args:
            data (pd.DataFrame): The DataFrame used to fit the scaler.
        """
        for column in self.columns:
            self.standardizers[column].fit(data[[column]])


class MissingValuesDropper(SimpleTransformation):
    """
    Drops rows with any missing values.
    """
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes any row containing at least one missing value.

        Args:
            data (pd.DataFrame): The DataFrame to be transformed.

        Returns:
            pd.DataFrame: The DataFrame with missing values dropped.
        """
        df = data.copy()
        return df.dropna()


class DuplicatesDropper(SimpleTransformation):
    """
    Drops duplicate rows from a DataFrame.
    """
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicate rows.

        Args:
            data (pd.DataFrame): The DataFrame to be transformed.

        Returns:
            pd.DataFrame: The DataFrame with duplicate rows removed.
        """
        df = data.copy()
        return df.drop_duplicates()


class NegativeDropper(SimpleTransformation):
    def __init__(self, columns: list):
        self.columns = columns

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df: pd.DataFrame = data.copy()

        for column in self.columns:
            df = df.drop(df[df[column] < 0].index)

        return df


class ColumnDropper(SimpleTransformation):
    """
    Drops specified columns from a DataFrame.

    Args:
        columns (list): A list of column names to drop.
    """
    def __init__(self, columns: list):
        self.columns = columns

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes specified columns from the DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to be transformed.

        Returns:
            pd.DataFrame: The DataFrame with specified columns dropped.
        """
        df: pd.DataFrame = data.copy()
        return df.drop(columns=self.columns)


class DatatypesTransformer(SimpleTransformation):
    """
    Converts the data types of specified columns.

    Args:
        map (dict[str, object]): A dictionary mapping column names to their new data types.
    """
    def __init__(self, map: dict[str, object]):
        self.map = map

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies data type conversion to specified columns.

        Args:
            data (pd.DataFrame): The DataFrame to be transformed.

        Returns:
            pd.DataFrame: The DataFrame with updated data types.
        """
        df = data.copy()

        for column in self.map:
            df[column] = df[column].astype(self.map[column])

        return df


class OneHotEncoder(SimpleTransformation):
    """
    Performs one-hot encoding on specified categorical columns.

    Args:
        columns (list): A list of column names to encode.
    """
    def __init__(self, columns: list):
        self.columns = columns

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies one-hot encoding to the specified columns.

        Args:
            data (pd.DataFrame): The DataFrame to be transformed.

        Returns:
            pd.DataFrame: The DataFrame with encoded columns.
        """
        df = data.copy()
        df = pd.get_dummies(df, columns=self.columns, drop_first=True)

        return df


class OutlierDropper(SimpleTransformation):
    """
    Drops rows identified as outliers by a specified outlier marker.

    Args:
        outliers_marker (BaseOutlierMarker): An object that can mark outliers.
    """
    def __init__(self, outliers_marker: BaseOutlierMarker):
        self.outliers_marker = outliers_marker

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Marks and removes rows identified as outliers.

        Args:
            data (pd.DataFrame): The DataFrame to be transformed.

        Returns:
            pd.DataFrame: The DataFrame with outlier rows removed.
        """
        df = data.copy()
        df, column = self.outliers_marker.mark(df)
        df = df[~df[column]]

        return df


class CSVExtractor(BaseExtractor):
    """
    Extracts data from a CSV file.

    Args:
        source (str): The file path of the CSV file.
    """
    def __init__(self, source: str):
        self.source = source

    def extract(self) -> pd.DataFrame:
        """
        Loads a CSV file into a pandas DataFrame.

        Raises:
            Exception: If the file does not exist or has an unsupported extension.

        Returns:
            pd.DataFrame: The extracted data.
        """
        assert os.path.exists(self.source), Exception(f"There is no file named '{self.source}'.")
        assert os.path.splitext(self.source)[1] == ".csv", Exception(f"Unsupported file extension '{self.source}'.")

        row_data = pd.read_csv(self.source)
        return row_data


class Transformer(BaseTransformer):
    """
    A data transformation pipeline orchestrator.

    It manages a sequence of transformations, handling both fitting and
    transforming steps.

    Args:
        transformations (list[SimpleTransformation | FittedTransformation]):
            A list of transformation objects to be applied in order.
    """
    def __init__(self, transformations: list[SimpleTransformation | FittedTransformation]):
        self.transformations: list[SimpleTransformation] = transformations

    def transform(self, row_data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a series of transformations to the data.

        This method assumes that any fitted transformations have already been fitted.

        Args:
            row_data (pd.DataFrame): The DataFrame to be transformed.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        assert row_data.shape[0] > 0, Exception("There is no row data with size > 0.")

        df: pd.DataFrame = row_data.copy()
        for transformation in self.transformations:
            df = transformation.transform(df)

        return df

    def fit(self, data: pd.DataFrame):
        """
        Orchestrates the fitting of transformations that require it.

        This method first applies simple transformations to clean the data,
        and then fits the transformations that require learning parameters.

        Args:
            data (pd.DataFrame): The DataFrame used to fit the transformations.
        """
        assert data.shape[0] > 0, Exception("There is no row data with size > 0.")

        df: pd.DataFrame = data.copy()

        for transformation in self.transformations:
            if isinstance(transformation, SimpleTransformation):
                df = transformation.transform(df)

        for transformation in self.transformations:
            if isinstance(transformation, FittedTransformation):
                transformation.fit(df)


class CSVLoader(BaseLoader):
    """
    Loads a DataFrame to a CSV file.

    Args:
        source (str): The file path to save the CSV file.
    """
    def __init__(self, source: str):
        self.source = source

    def load(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Saves a DataFrame to a specified CSV file.

        Args:
            data (pd.DataFrame): The DataFrame to be saved.
        """
        data.to_csv(self.source, index=False)

        return data


class ETL:
    """
    An ETL (Extract, Transform, Load) pipeline orchestrator.

    It manages the entire data flow from extraction to loading.

    Args:
        extractor (BaseExtractor): An object to extract data.
        transformer (BaseTransformer): An object to manage data transformation.
        loader (BaseLoader): An object to load the transformed data.
    """
    def __init__(self, extractor: BaseExtractor, transformer: BaseTransformer, loader: BaseLoader):
        self.extractor = extractor
        self.transformer = transformer
        self.loader = loader

    def process(self) -> pd.DataFrame:
        """
        Executes the full ETL pipeline.

        The pipeline extracts data, fits and transforms it, and then loads it.
        """
        row_data = self.extractor.extract()
        self.transformer.fit(row_data)
        processed_data = self.transformer.transform(row_data)
        self.loader.load(processed_data)

        return processed_data

