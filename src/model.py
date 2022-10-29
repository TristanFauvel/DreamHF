# This is the model submitted for evaluation in the challenge
from preprocessing import load_data, Salosensaari_processing
from pipeline import postprocessing, create_pipeline
from candidate_models import EarlyStoppingMonitor
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# %%
import sys
import os

arguments = sys.argv

try:
    root = arguments[1]
except:
    raise ValueError("You must provide the input path")

#%%


# Load the data
os.environ["root_folder"] = root

#%%
print("Processing the data...")
pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test = load_data(root)

X_train, X_test, y_train, y_test, test_sample_ids = Salosensaari_processing(
    pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test
)
"""
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
"""

monitor = EarlyStoppingMonitor(25, 50)
est_early_stopping = GradientBoostingSurvivalAnalysis()
pipe = create_pipeline(est_early_stopping)
pipe.fit = lambda X_train, y_train: pipe.fit(X_train, y_train, model__monitor=monitor)

distributions = dict(
    model__learning_rate=uniform(loc=0, scale=1),
    model__max_depth=randint(1, 4),
    model__loss=["coxph"],
    model__n_estimators=uniform(loc=30, scale=150),
    model__min_samples_split=randint(2, 10),
    model__min_samples_leaf=randint(1, 10),
    model__subsample=uniform(loc=0.5, scale=0.5),
    model__max_leaf_nodes=randint(2, 10),
    model__dropout_rate=uniform(loc=0, scale=1),
)


randsearchcv = RandomizedSearchCV(
    pipe, distributions, random_state=0, n_iter=300, n_jobs=-1
)
search = randsearchcv.fit(X_train, y_train)
best_model = search.best_estimator_
best_monitor = monitor


#%%

print("Prediction with the best model...")
best_model.fit(X_train, y_train, model__monitor=best_monitor)

# Return predictions
preds_test = best_model.predict(X_test)
postprocessing(preds_test, test_sample_ids)

print("Task completed.")
# %%
