import os

import neptune
import pandas as pd
import numpy as np
from hyperopt import hp
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
from ds_package.validation import validate
from ds_package.features import extract_features
from ds_package.hyperopt import get_best_hyperparameters
from airflow.sdk import Variable, dag, task
from airflow.providers.standard.operators.bash import BashOperator


@dag("mlops_experiment", start_date=pd.to_datetime('2023-01-01'), schedule=None, catchup=False)
def experiment():
    DVC_REPO_ROOT = "/opt/airflow/dvc_repo"
    DATA_FOLDER = f"{DVC_REPO_ROOT}/data"

    search_space = {
        "n_estimators": hp.quniform("n_estimators", 100, 500, 1),
        "max_depth": hp.quniform("max_depth", 1, 10, 1),
        "learning_rate": hp.loguniform("learning_rate", -4 * np.log(10), 2 * np.log(10))
    }

    load_data_from_dvc = BashOperator(
        task_id="dvc_pull_data",
        bash_command="dvc pull --force",
        cwd=DVC_REPO_ROOT,
        append_env=True,
    )

    @task
    def validate_dataset():
        raw_data_path = f"{DATA_FOLDER}/sales_post_process.csv"
        print(f"Reading dataset from: {raw_data_path}")
        dataset = pd.read_csv(raw_data_path, parse_dates=["date"], date_format="%Y-%m-%d")
        validate(dataset)


    @task
    def extract_features_from_data():
        raw_data_path = f"{DATA_FOLDER}/sales_post_process.csv"
        dataset = pd.read_csv(raw_data_path, parse_dates=["date"], date_format="%Y-%m-%d")
        data = extract_features(dataset)

        path = f"{DATA_FOLDER}/dataset.parquet"
        data.to_parquet(path, index=False)

        return path


    @task
    def train_test_split(path: str):
        df = pd.read_parquet(path)

        test = df[(df["year"] == 2015) & (df["month"] > 8)]
        train = df.drop(test.index)

        x_train, x_test = train.drop(columns=["item_cnt_day_lag_-31"]), test.drop(columns=["item_cnt_day_lag_-31"])
        y_train, y_test = train["item_cnt_day_lag_-31"], test["item_cnt_day_lag_-31"]

        paths = {
            "x_train": f"{DATA_FOLDER}/x_train.pkl",
            "y_train": f"{DATA_FOLDER}/y_train.pkl",
            "x_test": f"{DATA_FOLDER}/x_test.pkl",
            "y_test": f"{DATA_FOLDER}/y_test.pkl",
        }

        x_train.to_pickle(paths["x_train"])
        y_train.to_pickle(paths["y_train"])
        x_test.to_pickle(paths["x_test"])
        y_test.to_pickle(paths["y_test"])

        return paths

    @task
    def hyperparameter_search(paths: dict[str, str], search_space: dict):
        model = XGBRegressor()
        x_train = pd.read_pickle(paths["x_train"])
        y_train = pd.read_pickle(paths["y_train"])

        best = get_best_hyperparameters(
            search_space,
            model,
            x_train,
            y_train,
            50
        )

        return best

    @task
    def train_and_test(paths: dict[str, str], best: dict, search_space: dict):
        NEPTUNE_API_TOKEN = Variable.get("NEPTUNE_API_TOKEN", default=None)
        NEPTUNE_PROJECT = Variable.get("NEPTUNE_PROJECT", default=None)

        run = neptune.init_run(
            api_token=NEPTUNE_API_TOKEN,
            project=NEPTUNE_PROJECT
        )

        run["dataset_path"] = f"{DATA_FOLDER}/sales_post_process.csv"
        run["model"] = "XGBRegressor"
        run["hyperparameters_space"] = search_space
        run["best_hyperparameters"] = best

        best["n_estimators"] = int(best["n_estimators"])
        best["max_depth"] = int(best["max_depth"])

        x_train, y_train = pd.read_pickle(paths["x_train"]), pd.read_pickle(paths["y_train"])
        x_test, y_test = pd.read_pickle(paths["x_test"]), pd.read_pickle(paths["y_test"])

        model = XGBRegressor()
        model.set_params(**best)
        model.fit(x_train, y_train)

        test_predictions = model.predict(x_test)
        run["test_score"] = root_mean_squared_error(y_test, test_predictions)

        run.stop()

    load_data_task = load_data_from_dvc

    validation_instance = validate_dataset()
    processed_path_task = extract_features_from_data()

    load_data_task >> validation_instance
    load_data_task >> processed_path_task

    split_paths = train_test_split(path=processed_path_task)

    best_params = hyperparameter_search(paths=split_paths, search_space=search_space)

    train_and_test(paths=split_paths, best=best_params, search_space=search_space)


experiment()