# %%
import pandas as pd
import numpy as np
from sksurv.column import encode_categorical


def pheno_processing_pipeline(df, training):
    df = df.convert_dtypes()

    if "Event" in df:
        df.dropna(subset=["Event"], inplace=True)
        df = df.astype({"Event": "bool"})

    if "Event_time" in df:
        df.dropna(subset=["Event_time"], inplace=True)
        df = df.astype({"Event_time": "float64"})

    """   
    df = df.astype({'Age':'float64', 'Smoking':'bool', 'PrevalentCHD':'bool', 'BPTreatment':'bool', 'PrevalentDiabetes':'bool', 'PrevalentHFAIL':'bool',
                                       'Event':'bool', 'Sex':'bool', 'BodyMassIndex':'float64', 'SystolicBP':'float64', 'NonHDLcholesterol':'float64'})      
     
    """
    df.set_index("Unnamed: 0", inplace=True)

    df = df.rename_axis(index=None, columns=df.index.name)

    if training:
        artifacts = (df["Event_time"] < 0) & (df["Event"] == 1)
        df = df.loc[~artifacts, :]

    #############################################

    # selection = ((df.loc[:,'Event_time']>=0) & (((df.loc[:,'Event_time']<14.5) &  (df.loc[:,'Event']==1)) | (df.loc[:,'Event']==0)))
    # df = df.loc[selection,:]
    #############################################
    return df


def readcounts_processing_pipeline(df):
    df = df.transpose()
    df.columns = df.iloc[0]
    df = df.drop(labels=["Unnamed: 0"], axis=0)
    df = df.astype(np.int64)
    return df


def remove_unique_columns(df_train, df_test):
    for col in df_test.columns:
        if len(df_test[col].unique()) == 1 and len(df_train[col].unique()) == 1:
            df_test.drop(col, inplace=True, axis=1)
            df_train.drop(col, inplace=True, axis=1)
    return df_train, df_test


def load_data(root):
    pheno_df_train = pd.read_csv(root + "/train/pheno_training.csv")
    pheno_df_train = pheno_processing_pipeline(pheno_df_train, training=True)

    pheno_df_test = pd.read_csv(root + "/test/pheno_test.csv")
    pheno_df_test = pheno_processing_pipeline(pheno_df_test, training=False)

    readcounts_df_train = pd.read_csv(root + "/train/readcounts_training.csv")
    readcounts_df_train = readcounts_processing_pipeline(readcounts_df_train)

    readcounts_df_test = pd.read_csv(root + "/test/readcounts_test.csv")
    readcounts_df_test = readcounts_processing_pipeline(readcounts_df_test)

    readcounts_df_train, readcounts_df_test = remove_unique_columns(
        readcounts_df_train, readcounts_df_test
    )

    return pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test


def prepare_train_test(df_train, df_test, covariates):

    # Left truncation : we remove all participants who experienced HF before entering the study.

    selection_train = df_train.loc[:, "Event_time"] >= -np.inf #0
    
    test_sample_ids = df_test.index

    #Make sure that the features do not contain Event or Event_time
    if "Event" in covariates or "Event_time" in covariates:
        Exception("Event or Event_time are included in covariates, please remove them.")
      
    X_train = df_train.loc[selection_train, covariates]
    X_test = df_test.loc[:, covariates]
    y_train = df_train.loc[selection_train, ["Event", "Event_time"]]
    y_train = y_train.to_records(index=False)
     
    if "Event" in df_test:
        y_test = df_test.loc[:, ["Event", "Event_time"]]
        y_test = y_test.to_records(index=False)
    else:
        y_test = None

    return X_train, X_test, y_train, y_test, test_sample_ids


def check_data(df):
    # Check that the input data do not contain NaN
    nan_cols = df.isnull().values.any(axis=0)
    nan_counts = df.isnull().values.sum(axis=0)
    for nan, column, nan_c in zip(nan_cols, df.columns, nan_counts):
        if nan:
            print(f"Column {column} has {nan_c} missing values")

    n_deleted = df.shape[0] - df.dropna().shape[0]
    if n_deleted > 0:
        df.dropna(inplace=True)
        print(f"Deleted {n_deleted} rows with missing values")
    else:
        print(f"Number of rows with missing values: {n_deleted}")
        print("Please provide an imputation method")
    return df
