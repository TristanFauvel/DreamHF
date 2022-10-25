# %%
import pandas as pd
import numpy as np
import os
import pathlib

from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.utils import estimator_html_repr


def create_pipeline(model):
    def bool2cat(df):
        return df.astype("category")

    def cat2bool(df):
        return df.astype("bool")

    bool2cat = FunctionTransformer(bool2cat)
    cat2bool = FunctionTransformer(cat2bool)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("bool2cat", bool2cat),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("cat2bool", cat2bool),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, selector(dtype_exclude="bool")),
            ("cat", categorical_transformer, selector(dtype_include="bool")),
        ]
    )
    regressor = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    with open("regressor.html", "w") as f:
        f.write(estimator_html_repr(regressor))
    return regressor


def postprocessing(preds_test, test_sample_ids):
    # Check that the predictions do not contain NaN, +inf or -inf
    if np.any(np.isnan(preds_test)) or np.any(np.isinf(preds_test)):
        raise ValueError("Predictions contain invalid values (NaN or inf)")

    # Save results
    root = os.environ.get("root_folder")
    results = pd.DataFrame({"Score": preds_test}, index=test_sample_ids)
    results.index.name = "SampleID"
    outdir = root + "/output/"
    p = pathlib.Path(outdir)
    p.mkdir(parents=True, exist_ok=True)

    results.to_csv(outdir + "score.csv")
