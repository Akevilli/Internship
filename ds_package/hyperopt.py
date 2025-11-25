from typing import Any
from functools import partial

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from hyperopt import fmin, tpe, Trials, STATUS_OK


def get_best_hyperparameters(
    search_space: dict[str, Any],
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    max_evals: int = 50,
):
    def objective(params, model, x_train, y_train):
        params["n_estimators"] = int(params["n_estimators"])
        params["max_depth"] = int(params["max_depth"])
        model.set_params(**params)

        tscv = TimeSeriesSplit(n_splits=5)
        score = cross_val_score(model, x_train, y_train, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        return {"loss": -score.mean(), "params": params, "status": STATUS_OK}


    trials = Trials()

    best = fmin(
        fn=partial(objective, model=model, x_train=x_train, y_train=y_train),
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        show_progressbar=True,
        rstate=np.random.default_rng(42)
    )

    return best