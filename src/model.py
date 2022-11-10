# This is the model submitted for evaluation in the challenge
# %%
# from preprocessing import load_data, Salosensaari_processing
import os
import sys

from pipeline import postprocessing
from preprocessing import Salosensaari_processing, load_data
from survival_models import sksurv_gbt, xgb_aft, xgb_optuna, xgbse_weibull

arguments = sys.argv


####
#arguments = [0, '/home/tristan/Desktop/Repos/DreamHF']

####
try:
    ROOT = arguments[1]
except NameError as path_not_provided:
    raise ValueError("You must provide the input path") from path_not_provided


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

X_train, X_test, y_train, y_test, test_sample_ids = Salosensaari_processing(
    pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test, clinical_covariates
)

# %%
print("Definition of the cross-validation pipeline...")


model = xgbse_weibull()
model = sksurv_gbt()  # OK
model = xgb_aft()  # OK
model = xgb_optuna()
print("Search for optimal hyperparameters...")
preds_test, model = model.model_pipeline(X_train, y_train, X_test)


################
# %%
score_training, score_test = model.evaluate(X_train, X_test, y_train, y_test)
print(score_training)
print(score_test)
# %%
# Return predictions
postprocessing(preds_test, test_sample_ids, ROOT)

# save the model to disk
filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))

print("Task completed.")

# %%
