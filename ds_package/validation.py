import pandas as pd


_schema = {
    "date": pd.DatetimeTZDtype(tz="Europe/Moscow"),
    "date_block_num": pd.Int32Dtype(),
    "shop_id": pd.Int32Dtype(),
    "item_id": pd.Int32Dtype(),
    "item_price": pd.Float32Dtype(),
    "item_cnt_day": pd.Int32Dtype(),
    "shop_name": pd.StringDtype(),
    "item_name": pd.StringDtype(),
    "item_category_id": pd.Int32Dtype(),
    "item_category_name": pd.StringDtype(),
    "cluster": pd.Int32Dtype(),
}

def validate(df: pd.DataFrame):

    assert df.isna().sum().sum() == 0, f"Data has missing values! {df.isna().sum()}"
    assert df.duplicated().sum() == 0, "Data has duplicated!"

    expected_columns = set(_schema.keys())
    extra_columns = df.columns - expected_columns
    assert extra_columns, f"There are extra columns in the data! {extra_columns}"

    missing_columns = expected_columns - df.columns
    assert missing_columns, f"There are missing columns in the data! {missing_columns}"

    actual_types = df.dtypes.apply(lambda x: x.name).to_dict()
    type_mismatches = {}
    for column, type in _schema.items():
        if column in df.columns:
            actual_type = actual_types[column]

            if actual_type != type:
                type_mismatches[column] = (actual_type, type)

    if type_mismatches:
        mismatch_details = "\n".join([
            f"- Column '{col}': Actual type '{actual[0]}' != Expected type '{actual[1]}'"
            for col, actual in type_mismatches.items()
        ])
        raise TypeError(mismatch_details)




