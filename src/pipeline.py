# %%
import os
import pathlib

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils import estimator_html_repr

sklearn.set_config(transform_output="pandas")


def postprocessing(preds_test, test_sample_ids, root):
    # Check that the predictions do not contain NaN, +inf or -inf
    if np.any(np.isnan(preds_test)) or np.any(np.isinf(preds_test)):
        raise ValueError("Predictions contain invalid values (NaN or inf)")

    # Save results 
    results = pd.DataFrame({"Score": preds_test}, index=test_sample_ids)
    results.index.name = "SampleID"
    outdir = root + "/output/"
    p = pathlib.Path(outdir)
    p.mkdir(parents=True, exist_ok=True)

    results.to_csv(outdir + "scores.csv")


class EarlyStoppingMonitor:
    def __init__(self, window_size, max_iter_without_improvement):
        self.window_size = window_size
        self.max_iter_without_improvement = max_iter_without_improvement
        self._best_step = -1

    def __call__(self, iteration, estimator, args):
        # continue training for first self.window_size iterations
        if iteration < self.window_size:
            return False

        # compute average improvement in last self.window_size iterations.
        # oob_improvement_ is the different in negative log partial likelihood
        # between the previous and current iteration.
        start = iteration - self.window_size + 1
        end = iteration + 1
        improvement = np.mean(estimator.oob_improvement_[start:end])

        if improvement > 1e-6:
            self._best_step = iteration
            return False  # continue fitting

        # stop fitting if there was no improvement
        # in last max_iter_without_improvement iterations
        diff = iteration - self._best_step
        return diff >= self.max_iter_without_improvement


