from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import pandas as pd
import numpy as np

from pipeline import create_pipeline, EarlyStoppingMonitor
from scipy.stats import uniform, randint

monitor = EarlyStoppingMonitor(25, 50)
est_early_stopping = GradientBoostingSurvivalAnalysis()
pipe = create_pipeline(est_early_stopping)
pipe.fit = lambda X_train, y_train: pipe.fit(X_train, y_train, model__monitor=monitor)

distributions = dict(
    model__learning_rate=uniform(loc=0, scale=1),
    model__max_depth=randint(1, 4),
    model__loss=["coxph"],
    model__n_estimators=uniform(loc=30, scale=150),
    model__min_samples_split=randint(2, 10),
    model__min_samples_leaf=randint(1, 10),
    model__subsample=uniform(loc=0.5, scale=0.5),
    model__max_leaf_nodes=randint(2, 10),
    model__dropout_rate=uniform(loc=0, scale=1),
)

model = {
    "name": "Gradient Boosting sksurv",
    "pipe": pipe,
    "distributions": distributions,
    "monitor": monitor,
}
