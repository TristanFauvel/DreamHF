# This is the model submitted for evaluation in the challenge
# %%
# from preprocessing import load_data, Salosensaari_processing

import random
import sys
import warnings

from pipeline import experiment_pipeline
from preprocessing import load_data

random.seed(10)

arguments = sys.argv

# sklearn.set_config(transform_output="pandas")

# Checklist :
# - check submission name
# - set arguments
# - choose the model in pipeline
# - choose n_taxa and n_iter

############################################################
#arguments = [0, '/home/tristan/Desktop/Repos/DreamHF']
############################################################


# %%
SUBMISSION_NAME = "TristanF_Final_Submission"

SCORING = True

ROOT = arguments[1]


if ROOT == '/home/tristan/Desktop/Repos/DreamHF':
    warnings.warn(
        "Warning : the specified root directory is not compatible with code execution in a container environment")

print("Loading the data...")

pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test = load_data(
    ROOT, SCORING)

print("Data loaded. ")

N_TAXA = 0
N_ITER = 1000  # Number of hyperaparameters to test

# %%
experiment_pipeline(N_TAXA, N_ITER, pheno_df_train, pheno_df_test,
                    readcounts_df_train, readcounts_df_test, ROOT, SUBMISSION_NAME)


# %%
