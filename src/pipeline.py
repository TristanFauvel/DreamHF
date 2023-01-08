# %%
import os
import pathlib
import warnings

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
    xgb_optuna,
    xgbse_weibull,
)

sklearn.set_config(transform_output="pandas")

def postprocessing(preds, sample_ids, root, filename):
    # Check that the predictions do not contain NaN, +inf or -inf
    if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
        raise ValueError("Predictions contain invalid values (NaN or inf)")

    # Save results 
    results = pd.DataFrame({"Score": preds}, index=sample_ids)
    results.index.name = "SampleID"
    outdir = root + "/output/"
    p = pathlib.Path(outdir)
    p.mkdir(parents=True, exist_ok=True)

    results.to_csv(outdir + filename)



def test_output_csv(root, y_train, n_test):
    outdir = root + "/output/"
    risk_scores_train = pd.read_csv(outdir + "scores_train.csv")
    risk_scores_train = risk_scores_train.set_index('SampleID')
    harrell_C_training = concordance_index_censored(
        y_train['Event'], y_train['Event_time'], risk_scores_train.Score)[0]

    result = {
        "Harrell C": [harrell_C_training]
    }

    print(pd.DataFrame(result, index=["train"]))
    
    risk_scores = pd.read_csv(outdir + "scores.csv")
    risk_scores = risk_scores.set_index('SampleID')
    if any(risk_scores.Score < 0) or any(risk_scores.Score >1) or any(risk_scores.isna().Score):
         warnings.warn("Warning : Output file contains invalid values")

    if not (risk_scores.shape[0] == n_test):
       raise AssertionError("Incorrect number of test cases in the output file")



def experiment_pipeline(pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test, ROOT):

    processing = 'MI_clr'
    clinical_covariates = ['Age',
                           'BodyMassIndex',
                           'Smoking',
                           'BPTreatment',
                           'PrevalentDiabetes',
                           'PrevalentCHD',
                           #'PrevalentHFAIL',
                           'SystolicBP',
                           'NonHDLcholesterol',
                           'Sex']  # CLINICAL_COVARIATES
    n_taxa = 0
    
    if processing == 'Salosensaari':
        X_train, X_test, y_train, y_test, test_sample_ids, train_sample_ids = Salosensaari_processing(
            pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test, clinical_covariates
        )
    elif processing == 'MI_clr':
        ## Feature selection
        X_train, X_test, y_train, y_test, test_sample_ids, train_sample_ids = clr_processing(
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
    best_model = 'CoxPH'
    n_iter = 1000 #400
    
    n_test = pheno_df_test.shape[0]
    model = run_experiment(best_model, n_taxa, n_iter, X_train, X_test, y_train, test_sample_ids, train_sample_ids, ROOT, n_test)
    # %%
    print("Task completed.")
    return


def run_experiment(model_name, n_taxa, n_iter, X_train, X_test, y_train, test_sample_ids, train_sample_ids, ROOT, n_test):

    config = dict(
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

    model = eval(wandb_config.model_name + '(' + str(n_taxa) + ')')

    # %%
    print("Search for optimal hyperparameters...")
    model = model.cross_validation(X_train, y_train, n_iter)

    preds_test = model.risk_score(X_test)    
    postprocessing(preds_test, test_sample_ids, ROOT, "scores.csv")
                   
    preds_train = model.risk_score(X_train)    
    postprocessing(preds_train, train_sample_ids, ROOT, "scores_train.csv")
    
    test_output_csv(ROOT, y_train, n_test)
    
    run.finish()
    return model

# %%
