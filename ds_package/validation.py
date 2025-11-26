import pandas as pd


_schema = {
    "date": "datetime64[ns]",
    "date_block_num": "int64",
    "shop_id": "int64",
    "item_id": "int64",
    "item_price": "float64",
    "item_cnt_day": "int64",
    "shop_name": "object",
    "item_name": "object",
    "item_category_id": "int64",
    "item_category_name": "object",
    "cluster": "int64",
}

def validate(df: pd.DataFrame):

    assert df.isnull().sum().sum() == 0, f"Data has missing values! {df.isna().sum()}"
    assert df.duplicated().sum() == 0, "Data has duplicated!"

    expected_columns = set(_schema.keys())
    actual_columns = set(df.columns)
    extra_columns = actual_columns - expected_columns
    assert not extra_columns, f"There are extra columns in the data! {extra_columns}"

    missing_columns = expected_columns - actual_columns
    assert not missing_columns, f"There are missing columns in the data! {missing_columns}"

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




