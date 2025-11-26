import os
import warnings

import neptune
import pandas as pd
import numpy as np
from hyperopt import hp
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
from ds_package.validation import validate
from ds_package.features import extract_features
from ds_package.hyperopt import get_best_hyperparameters

warnings.filterwarnings('ignore')

# NEPTUNE_API_TOKEN = os.environ.get("NEPTUNE_API_TOKEN")
# NEPTUNE_PROJECT = os.environ.get("NEPTUNE_PROJECT")
#
# run = neptune.init_run(
#     api_token=NEPTUNE_API_TOKEN,
#     project=NEPTUNE_PROJECT
# )

search_space = {
    "n_estimators": hp.quniform("n_estimators", 100, 500, 1),
    "max_depth": hp.quniform("max_depth", 1, 10, 1),
    "learning_rate": hp.loguniform("learning_rate", -4*np.log(10), 2*np.log(10))
}
dataset = pd.read_csv("./data/sales_post_process.csv", parse_dates=["date"], date_format="%Y-%m-%d")
model = XGBRegressor()

# run["dataset_path"] = "./data/sales_post_process.csv"
# run["model"] = "XGBRegressor"
# run["hyperparameters_space"] = search_space

validate(dataset)
preprocessed_dataset = extract_features(dataset)

test = preprocessed_dataset[(preprocessed_dataset["year"] == 2015) & (preprocessed_dataset["month"] > 8)]
train = preprocessed_dataset.drop(test.index)

x_train, x_test = train.drop(columns=["item_cnt_day_lag_-31"]), test.drop(columns=["item_cnt_day_lag_-31"])
y_train, y_test = train["item_cnt_day_lag_-31"], test["item_cnt_day_lag_-31"]

best = get_best_hyperparameters(
    search_space=search_space,
    model=model,
    x_train=x_train,
    y_train=y_train,
    max_evals=50
)

# run["best_hyperparameters"] = best
best["n_estimators"] = int(best["n_estimators"])
best["max_depth"] = int(best["max_depth"])
model.set_params(**best)
model.fit(x_train, y_train)

test_predictions = model.predict(x_test)
# run["test_score"] = root_mean_squared_error(y_test, test_predictions)
# run.stop()