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

############################################################
arguments = [0, '/home/tristan/Desktop/Repos/DreamHF']
############################################################


ROOT = arguments[1]


if ROOT == '/home/tristan/Desktop/Repos/DreamHF':
    warnings.warn("Warning : the specified root directory is not compatible with code execution in a container environment")

print("Loading the data...")
pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test = load_data(
    ROOT)
experiment_pipeline(pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test, ROOT)


# %%
