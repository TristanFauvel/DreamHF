# %%
import pandas as pd
import numpy as np
import os
import pathlib


def postprocessing(preds_test, test_sample_ids):
    # Check that the predictions do not contain NaN, +inf or -inf   
    if np.any(np.isnan(preds_test)) or np.any(np.isinf(preds_test)):
        raise ValueError('Predictions contain invalid values (NaN or inf)')
    
    # Save results
    root = os.environ.get("root_folder")
    results = pd.DataFrame({"Score": preds_test}, index=test_sample_ids)
    results.index.name = "SampleID"
    outdir = root + "/output/"
    p = pathlib.Path(outdir)
    p.mkdir(parents=True, exist_ok=True)

    results.to_csv(outdir + "score.csv")
