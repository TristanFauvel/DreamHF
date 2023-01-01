# %%
import os
import pathlib

import numpy as np
import pandas as pd
import sklearn
from sksurv.metrics import concordance_index_censored

import wandb
from model_evaluation import evaluate_model
from preprocessing import CLINICAL_COVARIATES, Salosensaari_processing, clr_processing
from survival_models import (
    Coxnet,
    CoxPH,
    IPCRidge_sksurv,
    sksurv_gbt,
    sksurv_gbt_optuna,
    xgb_aft,
    xgb_optuna,
    xgbse_weibull,
)

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


def experiment_pipeline(pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test, ROOT):

    processing = 'MI_clr'
    clinical_covariates=CLINICAL_COVARIATES
    n_taxa = 50
    
    if processing == 'Salosensaari':
        X_train, X_test, y_train, y_test, test_sample_ids = Salosensaari_processing(
            pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test, clinical_covariates
        )
    elif processing == 'MI_clr':
        ## Feature selection
        X_train, X_test, y_train, y_test, test_sample_ids = clr_processing(
            pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test, clinical_covariates,  n_taxa)

    #%%
    """
    candidate_models = ['Coxnet', 'sksurv_gbt',
                        'xgb_optuna', 'CoxPH']

    best_perf = 0
    best_model = None

    for model_name in candidate_models:
        model = run_experiment(model_name, 50, X_train, X_test, y_train, y_test, test_sample_ids, ROOT)
        if model.harrell_C_test > best_perf:
            best_perf = model.harrell_C_test
            best_model = model_name
            # save the model to disk
            #filename = 'trained_model.sav'
            #pickle.dump(model, open(filename, 'wb'))
    """
    best_model = 'sksurv_gbt'
    model = run_experiment(best_model, 10, X_train, X_test, y_train, y_test, test_sample_ids, ROOT)
    # %%
    print("Task completed.")
    return


def run_experiment(model_name, n_iter, X_train, X_test, y_train, y_test, test_sample_ids, ROOT):

    config = dict(
        with_pca=False,
        # xgbse_weibull, sksurv_gbt, xgb_aft, xgb_optuna, CoxPH
        #  xgbse_weibull : not ok
        # Coxnet : ok, IPCRidge_sksurv
        #  CoxPH : ok, sksurv_gbt : ok, xgb_aft: ok, but severe overfitting (0.9676036909132226, 0.6376541333063073)
        model_name=model_name, 
        n_iter=n_iter,
    )

    run = wandb.init(
        project="Dream-Challenge",
        name=config['model_name'],
        notes="Compare models",
        tags=["baseline"],
        config=config,
        mode="disabled"  # disabled
    )

    # Load the data
    os.environ["root_folder"] = ROOT

    print("Processing the data...")

    wandb_config = wandb.config 

    
    #model = eval(wandb_config.model_name +                  f'(with_pca = {wandb_config.with_pca}, n_components = {wandb_config.n_components})')

    model = eval(wandb_config.model_name)

    # %%
    print("Search for optimal hyperparameters...")
    model = model.model_pipeline(X_train, y_train, wandb_config.n_iter)

    #%%
    model = model.evaluate(X_train, X_test, y_train, y_test)
    
    #%%
    wandb.log({'Harrel C - training':  model.harrell_C_training,
               'Harrel C - test':  model.harrell_C_test,
               'config':  config})
    #'HL - training': self.HL_training,
    #'HL - test': self.HL_test})

    wandb.run.summary['Harrel C - training'] = model.harrell_C_training
    wandb.run.summary['Harrel C - test'] = model.harrell_C_test

    preds_test = model.risk_score(X_test)
    postprocessing(preds_test, test_sample_ids, ROOT)
    run.finish()
    return model

# %%
