import pandas as pd
import numpy as np

from xgbse import XGBSEStackedWeibull
from sklearn.pipeline import create_pipeline, EarlyStoppingMonitor
from scipy.stats import uniform, randint

monitor = EarlyStoppingMonitor(25, 50)


xgb_params = {
    "aft_loss_distribution": "normal",
    "aft_loss_distribution_scale": 1,
    "booster": "dart",
    "colsample_bynode": 0.5,
    "eval_metric": "aft-nloglik",
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 50,
    "objective": "survival:aft",
    "subsample": 0.5,
    "tree_method": "hist",
}
# fitting with early stopping
xgb_model = XGBSEStackedWeibull(xgb_params)

pipe = create_pipeline(xgb_model)

pipe.fit = lambda X_train, y_train: pipe.fit(X_train, y_train, model__monitor=monitor)

distributions = dict(
    model__aft_loss_distribution_scale=uniform(loc=0, scale=1),
    model__booster=randint(1, 4),
    model__aft_loss_distribution=["coxph"],
    model__colsample_bynode=uniform(loc=30, scale=150),
    model__eval_metric=randint(2, 10),
    model__learning_rate=randint(1, 10),
    model__max_depth=uniform(loc=0.5, scale=0.5),
    model__min_child_weight=randint(2, 10),
    model__objective=uniform(loc=0, scale=1),
    model__subsample=uniform(loc=0.5, scale=1),
    model__tree_method=["hist"],
)

model = {
    "name": "XGBSE Weibull",
    "pipe": pipe,
    "distributions": distributions,
    "monitor": None,
}
