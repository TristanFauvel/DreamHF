# This is the model submitted for evaluation in the challenge
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sklearn.model_selection import RandomizedSearchCV
from numpy.random import randint, uniform
from src.preprocessing import load_data, Salosensaari_processing
from src.pipeline import postprocessing, create_pipeline


# %%
import sys
import os

arguments = sys.argv

try:
    ROOT = arguments[1]
except NameError as path_not_provided: 
    raise ValueError("You must provide the input path") from path_not_provided

# %%


# Load the data
os.environ["ROOT_folder"] = ROOT

# %%
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

pheno_df_train = pheno_df_train.loc[clinical_covariates, :]
pheno_df_test = pheno_df_test.loc[clinical_covariates, :]

X_train, X_test, y_train, y_test, test_sample_ids = Salosensaari_processing(
    pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test
)


print("Definition of the cross-validation pipeline...")
monitor = EarlyStoppingMonitor(25, 50)
est_early_stopping = GradientBoostingSurvivalAnalysis()
pipe = create_pipeline(est_early_stopping)
pipe.fit = lambda X_train, y_train: pipe.fit(
    X_train, y_train, model__monitor=monitor)

distributions = dict(
    model__learning_rate=uniform(low=0, high=1),
    model__max_depth=randint(1, 4),
    model__loss=["coxph"],
    model__n_estimators=uniform(low=30, high=150),
    model__min_samples_split=randint(2, 10),
    model__min_samples_leaf=randint(1, 10),
    model__subsample=uniform(low=0.5, high=0.5),
    model__max_leaf_nodes=randint(2, 10),
    model__dropout_rate=uniform(low=0, high=1),
)

print("Search for optimal hyperparameters...")
randsearchcv = RandomizedSearchCV(
    pipe, distributions, random_state=0, n_iter=300, n_jobs=-1, verbose=2
)
search = randsearchcv.fit(X_train, y_train)
best_model = search.best_estimator_
best_monitor = monitor


# %%

print("Prediction with the best model...")
best_model.fit(X_train, y_train, model__monitor=best_monitor)

# Return predictions
preds_test = best_model.predict(X_test)
postprocessing(preds_test, test_sample_ids)

print("Task completed.")
# %%
