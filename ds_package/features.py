import pandas as pd
from sklearn.pipeline import Pipeline

from .Pipeline import (
    ConditionDropper,
    GroupAndSummarize,
    LagFeatureGenerator,
    NanDropper,
    MovingAverageGenerator,
    DifferenceGenerator,
    DateDivider
)

_pipeline = Pipeline([
    ("GroupAndSummarize1", GroupAndSummarize(group_by_col=["shop_id", "date", "item_id"], agg_dict={"item_cnt_day": "sum"})),
    ("ConditionDropper", ConditionDropper("item_cnt_day", lambda df: (df > 20) | (df < 0))),

    ("LagFeatureGenerator_target", LagFeatureGenerator(column="item_cnt_day", lags=[-31], group_by_col=["shop_id", "item_id"], sort_by_col=["date"])),
    ("NanDropper1", NanDropper(columns=["item_cnt_day_lag_-31"])),

    ("MovingAverageGenerator1", MovingAverageGenerator(column="item_cnt_day", window_size=30, min_periods=1, group_by_col=["shop_id", "item_id"], sort_by_col=["date"])),

    ("DifferenceGenerator1", DifferenceGenerator(columns=["item_cnt_day"], lag=1, count=1, group_by_col=["shop_id", "item_id"], sort_by_col=["date"])),
    ("NanDropper2", NanDropper(columns=["item_cnt_day_diff_lag1_c1"])),

    ("LagFeatureGenerator2", LagFeatureGenerator(column="item_cnt_day_diff_lag1_c1", lags=[1], group_by_col=["shop_id", "item_id"], sort_by_col=["date"])),
    ("NanDropper3", NanDropper(columns=["item_cnt_day_diff_lag1_c1_lag_1"])),

    ("MovingAverageGenerator2", MovingAverageGenerator(column="item_cnt_day_diff_lag1_c1", window_size=30, min_periods=1, group_by_col=["shop_id", "item_id"], sort_by_col=["date"])),

    ("DateDivider", DateDivider(column="date")),
    ("NanDropper4", NanDropper())
])

def extract_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    return _pipeline.fit_transform(df)