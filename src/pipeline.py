# %%
import os
import pathlib
import warnings

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored

import wandb
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


def postprocessing(preds, sample_ids, root, filename, submission_name):
    """
    Save the predictions in a CSV file with the given filename, in the output directory specified by the root path.
    The predictions will be saved in a single column named "Score", with the sample IDs as the index.
    If the root path is the DreamHF repository, the output directory will be "DreamHF/output/". Otherwise, the output
    directory will be "<root>/<submission_name>/output/".

    Args:
    - preds: numpy array of shape (n_samples,) with the risk scores predicted by the model
    - sample_ids: list of strings with the sample IDs corresponding to the predictions
    - root: string representing the root path where the output directory will be created
    - filename: string representing the name of the output CSV file
    - submission_name: string representing the name of the submission, which will be used to create the output directory

    Returns: None
    """
    # Check that the predictions do not contain NaN, +inf or -inf
    if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
        raise ValueError("Predictions contain invalid values (NaN or inf)")

    # Save results
    results = pd.DataFrame({"Score": preds}, index=sample_ids)
    results.index.name = "SampleID"

    if root == '/home/tristan/Desktop/Repos/DreamHF':
        outdir = root + "/output/"
    else:
        outdir = root + '/' + submission_name + "/output/"

    p = pathlib.Path(outdir)
    p.mkdir(parents=True, exist_ok=True)

    results.to_csv(outdir + filename)


def test_output_csv(root, y_train, n_test, submission_name):
    """
    Read the CSV files containing the risk scores for the training and test sets, and compute the Harrell's C-index
    for the training set. Raise an exception if the number of samples in the test set does not match the expected value
    n_test, or if the output file contains invalid values (negative or greater than 1, or NaN). Print the value of
    Harrell's C-index for the training set.

    Args:
    - root: string representing the root path where the output directory is located
    - y_train: pandas DataFrame containing the clinical data and survival information for the training set
    - n_test: integer representing the expected number of samples in the test set
    - submission_name: string representing the name of the submission, which will be used to locate the output directory

    Returns: None
    """
    if root == '/home/tristan/Desktop/Repos/DreamHF':
        outdir = root + "/output/"
    else:
        outdir = root + '/' + submission_name + "/output/"

    risk_scores_train = pd.read_csv(outdir + "scores_train.csv")
    risk_scores_train = risk_scores_train.set_index('SampleID')
    harrell_c_training = concordance_index_censored(
        y_train['Event'], y_train['Event_time'], risk_scores_train.Score)[0]

    result = {
        "Harrell C": [harrell_c_training]
    }

    print(pd.DataFrame(result, index=["train"]))

    risk_scores = pd.read_csv(outdir + "scores.csv")
    risk_scores = risk_scores.set_index('SampleID')
    if any(risk_scores.Score < 0) or any(risk_scores.Score > 1) or any(risk_scores.isna().Score):
        warnings.warn("Warning: Output file contains invalid values")

    if not (risk_scores.shape[0] == n_test):
        raise AssertionError(
            "Incorrect number of test cases in the output file")


def experiment_pipeline(n_taxa, n_iter, pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test, ROOT, submission_name):
    """
    Pipeline to run experiments on microbiome data and clinical covariates to predict patient survival using
    different models. 

    Args:
        n_taxa (int): number of taxa to use in feature selection
        n_iter (int): number of iterations for cross-validation
        pheno_df_train (pandas DataFrame): phenotype data for the training set
        pheno_df_test (pandas DataFrame): phenotype data for the test set
        readcounts_df_train (pandas DataFrame): microbiome read counts data for the training set
        readcounts_df_test (pandas DataFrame): microbiome read counts data for the test set
        ROOT (str): path to the root directory
        submission_name (str): name of the submission file

    Returns:
        None
    """
    processing = 'MI_clr'
    clinical_covariates = ['Age',
                           'BodyMassIndex',
                           'Smoking',
                           'BPTreatment',
                           'PrevalentDiabetes',
                           'PrevalentCHD',
                           # 'PrevalentHFAIL',
                           'SystolicBP',
                           'NonHDLcholesterol',
                           'Sex']  # CLINICAL_COVARIATES

    if processing == 'Salosensaari':
        # Run Salosensaari processing
        x_train, x_test, y_train, y_test, test_sample_ids, train_sample_ids = Salosensaari_processing(
            pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test, clinical_covariates
        )
    elif processing == 'MI_clr':
        # Run feature selection using clr_processing
        x_train, x_test, y_train, y_test, test_sample_ids, train_sample_ids = clr_processing(
            pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test, clinical_covariates,  n_taxa)
    else:
        raise ValueError('Invalid processing')

    # Choose the best model based on performance on the validation set
    best_model = 'CoxPH'

    n_test = pheno_df_test.shape[0]
    # Run the chosen model on the training and test data
    run_experiment(best_model, n_taxa, n_iter, x_train, x_test, y_train,
                   test_sample_ids, train_sample_ids, ROOT, n_test, submission_name)

    print("Task completed.")
    return


def run_experiment(model_name, n_taxa, n_iter, x_train, x_test, y_train, test_sample_ids, train_sample_ids, ROOT, n_test, submission_name):
    """
    Runs the selected model on the training and test data.

    Args:
        model_name (str): name of the model to use
        n_taxa (int): number of taxa to use in feature selection
        n_iter (int): number of iterations for cross-validation
        x_train (numpy ndarray): microbiome data for the training set
        x_test (numpy ndarray): microbiome data for the test set
        y_train (numpy ndarray): survival data for the training set
        test_sample_ids (list): list of sample IDs for the test set
        train_sample_ids (list): list of sample IDs for the training set
        ROOT (str): path to the root directory
        n_test (int): number of samples in the test set
        submission_name (str): name of the submission file

    Returns:
        model: the trained model
    """

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
    model = model.cross_validation(x_train, y_train, n_iter)

    preds_test = model.risk_score(x_test)
    postprocessing(preds_test, test_sample_ids, ROOT,
                   "scores.csv", submission_name)

    preds_train = model.risk_score(x_train)
    postprocessing(preds_train, train_sample_ids, ROOT,
                   "scores_train.csv", submission_name)

    test_output_csv(ROOT, y_train, n_test, submission_name)

    run.finish()
    return model

# %%
