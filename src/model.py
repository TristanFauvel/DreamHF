# This is the model submitted for evaluation in the challenge
# %%
import sys
import os

arguments = sys.argv

try:
    root = arguments[1]
except:
    raise Exception("You must provide the input path")

import numpy as np
from preprocessing import load_data, prepare_train_test

from pipeline import postprocessing, create_pipeline
from sklearn.model_selection import cross_val_score 
from candidate_models import candidate_models_df
from model_evaluation import evaluate_model

# Load the data
os.environ["root_folder"] = root


print("Processing the data...")
pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test = load_data(root)
 
 
df_train = pheno_df_train.join(readcounts_df_train)
df_test = pheno_df_test.join(readcounts_df_test)
covariates = df_train.loc[:, (df_train.columns != 'Event') & (df_train.columns != 'Event_time')].columns

X_train, X_test, y_train, y_test,test_sample_ids = prepare_train_test(df_train, df_test, covariates)

# Loop over the candidate models and optimize them individually
print("Crossvalidation of candidate models...")
best_score = 0
for index, row in candidate_models_df.iterrows():
    model = row["model_name"]
    monitor = row["est_monitor"]

    model = create_pipeline(model) 
    scores = cross_val_score(model, X_train, y_train, cv=5)
    score = np.mean(scores)
    if score > best_score:
        best_score = score
        best_model = model
        best_monitor = monitor

print("Prediction with the best model...")
best_model.fit(X_train, y_train, model__monitor=best_monitor)

# Return predictions
preds_test = best_model.predict(X_test)
postprocessing(preds_test, test_sample_ids)

print("Task completed.")
# %%
