# This is the model submitted for evaluation in the challenge
# %%
# from preprocessing import load_data, Salosensaari_processing
import os
import pickle
import random
import sys

from pipeline import postprocessing
from preprocessing import (
    CLINICAL_COVARIATES,
    Salosensaari_processing,
    clr_processing,
    load_data,
    taxa_selection,
)
from survival_models import CoxPH, sksurv_gbt, xgb_aft, xgb_optuna, xgbse_weibull

random.seed(10)

arguments = sys.argv
 
    
arguments = [0, '/home/tristan/Desktop/Repos/DreamHF']
ROOT = arguments[1]

print("Loading the data...")
pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test = load_data(
    ROOT)


import wandb

#%%

config = dict (
    with_pca = False,
    n_components = None,
    # xgbse_weibull, sksurv_gbt, xgb_aft, xgb_optuna
    #  CoxPH : ok, sksurv_gbt : ok
    model_name='xgbse_weibull',
    clinical_covariates = CLINICAL_COVARIATES,
    processing = 'MI_clr', 
    n_iter = 5,   
    n_taxa = 30
)

run = wandb.init(
project="Dream-Challenge",
name = config['model_name'],
notes="Compare models",
tags=["baseline"],
config=config,
mode="disabled" #disabled
)

# Load the data
os.environ["root_folder"] = ROOT


print("Processing the data...")


wandb_config = wandb.config
clinical_covariates = wandb_config.clinical_covariates

#%%
if wandb_config.processing == 'Salosensaari':
    X_train, X_test, y_train, y_test, test_sample_ids = Salosensaari_processing(
        pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test, clinical_covariates
    )
elif wandb_config.processing == 'MI_clr':
## Feature selection
    X_train, X_test, y_train, y_test, test_sample_ids = clr_processing(
        pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test, clinical_covariates, wandb_config.n_taxa)

model = eval(wandb_config.model_name + f'(with_pca = {wandb_config.with_pca}, n_components = {wandb_config.n_components})')
              
# %%

print("Search for optimal hyperparameters...")
model = model.model_pipeline(X_train, y_train, wandb_config.n_iter)

# %%
preds_test= model.risk_score(X_test)
postprocessing(preds_test, test_sample_ids, ROOT)


import matplotlib.pyplot as plt

plt.hist(preds_test)
model.estimator[:1].transform(X_test)

#%%
self = model.evaluate(X_train, X_test, y_train, y_test)
print(self.harrell_C_training)
print(self.harrell_C_test)
#%%
wandb.log({'Harrel C - training': self.harrell_C_training, 
            'Harrel C - test': self.harrell_C_test,
            'config':  config})
            #'HL - training': self.HL_training,
            #'HL - test': self.HL_test})  
            
wandb.run.summary['Harrel C - training'] =  self.harrell_C_training 
wandb.run.summary['Harrel C - test'] =  self.harrell_C_test 

# save the model to disk
filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))

print("Task completed.")
run.finish()

# %%
