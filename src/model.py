# This is the model submitted for evaluation in the challenge
# %%
import sys
import pandas as pd
import numpy as np
import pathlib
import sklearn 
from sksurv.functions import StepFunction
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.column import encode_categorical
from sksurv.ensemble import RandomSurvivalForest
 
from preprocessing import readcounts_processing_pipeline, pheno_processing_pipeline, prepare_train_test, remove_unique_columns
from HosmerLemeshow import HosmerLemeshow, reformat_inputs

arguments = sys.argv
  
root = arguments[1]

pheno_df_train = pd.read_csv(root + 'train/pheno_training.csv')
pheno_df_train = pheno_processing_pipeline(pheno_df_train)

pheno_df_test = pd.read_csv(root + 'test/pheno_test.csv')
pheno_df_test = pheno_processing_pipeline(pheno_df_test)

readcounts_df_train = pd.read_csv(root + 'train/readcounts_training.csv')
readcounts_df_train = readcounts_processing_pipeline(readcounts_df_train)

readcounts_df_test = pd.read_csv(root + 'test/readcounts_test.csv')
readcounts_df_test = readcounts_processing_pipeline(readcounts_df_test)
 
 
pheno_df = pd.concat([pheno_df_train, pheno_df_test])
pheno_df = pheno_df.loc[pheno_df.Event_time > 0]
t0 = pheno_df['Event_time'].min()
tf = pheno_df['Event_time'].max()

times = np.linspace(t0, tf, 15)
times = times[1:-1] 


base_model = CoxPHSurvivalAnalysis(alpha=0, ties='breslow', n_iter=100, tol=1e-09, verbose=0)
covariates = ['Sex=1', 'Age']
X_train, X_test, y_train, y_test, test_sample_ids = prepare_train_test(pheno_df_train, pheno_df_test, covariates)

"""
# Random forest survival model with all clinical covariates + microbiome data
readcounts_df_train, readcounts_df_test = remove_unique_columns(readcounts_df_train, readcounts_df_test)

 
df_train = pheno_df_train.join(readcounts_df_train)
df_test = pheno_df_test.join(readcounts_df_test)
base_model = RandomSurvivalForest(n_estimators=100, max_depth=None, min_samples_split=6, min_samples_leaf=3)  

covariates = df_train.columns                   
X_train, X_test, y_train, y_test,test_sample_ids = prepare_train_test(df_train, df_test, covariates)
"""


base_model.fit(X_train, y_train)

preds_test = base_model.predict(X_test)
 
results = pd.DataFrame({'Score':preds_test}, index = test_sample_ids)
results.index.name = 'SampleID'
 
Team_Name_Submission_Number = 'TristanF_Submission_1'

import pathlib

outdir = root + '/' + Team_Name_Submission_Number + '/output/'
p = pathlib.Path(outdir)
p.mkdir(parents=True, exist_ok=True)

results.to_csv(outdir + 'score.csv')

# %%
