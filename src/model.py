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
from survival_models import sksurv_gbt, xgb_aft, xgb_optuna, xgbse_weibull

random.seed(10)

arguments = sys.argv

scoring = False
if scoring:
    try:
        ROOT = arguments[1]
    except NameError as path_not_provided:
        raise ValueError("You must provide the input path") from path_not_provided
else:    
    
    arguments = [0, '/home/tristan/Desktop/Repos/DreamHF']
    
# Load the data
os.environ["root_folder"] = ROOT


print("Processing the data...")
pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test = load_data(
    ROOT)

clinical_covariates = [
    "Age",
    "BodyMassIndex",
    "Smoking",
    "BPTreatment",
    "SystolicBP",
    "NonHDLcholesterol",
]
clinical_covariates = CLINICAL_COVARIATES

#X_train, X_test, y_train, y_test, test_sample_ids = Salosensaari_processing(
#    pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test, clinical_covariates
#)

## Feature selection

selected_taxa = taxa_selection(pheno_df_train, readcounts_df_train)
df_train = pheno_df_train
selection = (df_train.columns != 'Event') & (df_train.columns != 'Event_time')
covariates = df_train.columns[selection].to_numpy()
X_train, X_test, y_train, y_test, test_sample_ids = clr_processing(
    pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test, covariates, selected_taxa)



# %%
print("Definition of the cross-validation pipeline...")

model = xgbse_weibull()
model = sksurv_gbt()  # OK
model = xgb_aft()  # OK
model = xgb_optuna()
model = sksurv_gbt(with_pca = False)  # OK with PCA, test : 0.6999662173575217
print("Search for optimal hyperparameters...")
model = model.model_pipeline(X_train, y_train)

# %%
preds_test= model.risk_score(X_test)
postprocessing(preds_test, test_sample_ids, ROOT)

if not scoring :
    import matplotlib.pyplot as plt
    plt.hist(preds_test)
    model.estimator[:1].transform(X_test)

    self = model.evaluate(X_train, X_test, y_train, y_test)
    print(self.score_training)
    print(self.score_test)
    
    # save the model to disk
    filename = 'trained_model.sav'
    pickle.dump(model, open(filename, 'wb'))

print("Task completed.")


# %%
