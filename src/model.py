# This is the model submitted for evaluation in the challenge
# %%
# from preprocessing import load_data, Salosensaari_processing
 
import random
import sys

from pipeline import experiment_pipeline
from preprocessing import load_data

random.seed(10)

arguments = sys.argv

ROOT = arguments[1]

print("Loading the data...")
pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test = load_data(
    ROOT)
experiment_pipeline(pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test, ROOT)


# %%
