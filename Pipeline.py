from typing import Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Pipeline:
    def __init__(self, steps: list[tuple[str, Union[BaseEstimator, TransformerMixin], bool]]):
        """
        Initializes the pipeline with a list of steps and a separate model.

        Args:
            steps (list[tuple[str, object, bool]]): A list of tuples, where each tuple
                contains a step's name, the step object (transformer), and a boolean
                flag to skip it during prediction.
            model: The final model to be trained and used for prediction.
        """

        self.steps = steps

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the pipeline's steps and the final model.

        Args:
            X (pd.DataFrame): The training data.
            y (pd.Series): The target labels.
        """
        X_processed = X.copy()
        for name, step, _ in self.steps:
            if hasattr(step, 'fit_transform'):
                X_processed = step.fit_transform(X_processed, y)
            elif hasattr(step, 'fit') and hasattr(step, 'transform'):
                step.fit(X_processed, y)
                X_processed = step.transform(X_processed)
            elif hasattr(step, 'transform'):
                X_processed = step.transform(X_processed)
            else:
                raise AttributeError(f"Step '{name}' has no valid fit/transform methods.")

    def transform(self, X: pd.DataFrame):
        """
        Applies transformations and uses the trained model to make predictions.

        Args:
            X (pd.DataFrame): The data to make predictions on.
        """
        X_processed = X.copy()
        for name, step, skip in self.steps:
            if skip:
                continue

            if hasattr(step, 'transform'):
                X_processed = step.transform(X_processed)
            else:
                raise AttributeError(f"Step '{name}' has no 'transform' method.")

        return X_processed


class GroupAndSummarize(BaseEstimator, TransformerMixin):
    """
    A custom transformer to group data, summarize columns, and add new ones.

    This class is designed to be a step in a scikit-learn compatible pipeline.
    """

    def __init__(self, group_by_col: list[str], agg_dict: dict):
        """
        Initializes the transformer with grouping and aggregation parameters.

        Args:
            group_by_col (str): The column name to group by.
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