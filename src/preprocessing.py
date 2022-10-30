# %%
import pandas as pd
import numpy as np
from sksurv.column import encode_categorical
from skbio.stats.composition import multiplicative_replacement
from skbio.stats.composition import clr
import xgboost as xgb


def pheno_processing_pipeline(df, training):
    df = df.convert_dtypes()

    if "Event" in df:
        df.dropna(subset=["Event"], inplace=True)
        df = df.astype({"Event": "bool"})

    if "Event_time" in df:
        df.dropna(subset=["Event_time"], inplace=True)
        df = df.astype({"Event_time": "float64"})

    df.set_index("Unnamed: 0", inplace=True)

    df = df.rename_axis(index=None, columns=df.index.name)

    if training:
        artifacts = (df["Event_time"] < 0) & (df["Event"] == 1)
        df = df.loc[~artifacts, :]

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

    idx_pheno_train = pheno_df_train.index
    idx_pheno_test = pheno_df_test.index
    idx_read_train = readcounts_df_train.index
    idx_read_test = readcounts_df_test.index

    idx_train = idx_pheno_train.intersection(idx_read_train)
    idx_test = idx_pheno_test.intersection(idx_read_test)

    readcounts_df_train = readcounts_df_train.loc[idx_train, :]
    readcounts_df_test = readcounts_df_test.loc[idx_test, :]
    pheno_df_test = pheno_df_test.loc[idx_test, :]
    pheno_df_train = pheno_df_train.loc[idx_train, :]

    return pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test


def prepare_train_test(df_train, df_test, covariates):
    # Left truncation : we remove all participants who experienced HF before entering the study.
    selection_train = df_train.loc[:, "Event_time"] >= -np.inf  # 0

    test_sample_ids = df_test.index

    # Make sure that the features do not contain Event or Event_time
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


def taxa_aggregation(readcounts_df, taxonomic_level="s__"):
    # Aggregate the species into genus
    readcounts_df.columns = [
        el.split(taxonomic_level)[0] for el in readcounts_df.columns
    ]
    readcounts_df = readcounts_df.groupby(readcounts_df.columns, axis=1).sum()
    return readcounts_df


def taxa_filtering(readcounts_df):
    ## Select genus-level taxonomic groups that were detected in >1% of the study participants at a within-sample relative abundance of >0.1%.
    total = readcounts_df.sum(axis=1)
    df_proportions = readcounts_df.divide(total, axis="rows")
    selection = (df_proportions > 0.001).mean(axis=0) > 0.01
    readcounts_df = readcounts_df.loc[:, selection]

    # Median relative abundance of the selected genus
    relative_abundance = df_proportions.loc[:, selection].sum(axis=1)
    relative_abundance.median()

    return selection


def centered_log_transform(readcounts_df):
    ## Centered log transformation
    X_mr = multiplicative_replacement(readcounts_df)

    # CLR
    X_clr = clr(X_mr)

    df = pd.DataFrame(X_clr, columns=readcounts_df.columns, index=readcounts_df.index)
    return df


def Salosensaari_processing(
    pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test
):
    readcounts_df_train = taxa_aggregation(readcounts_df_train)
    selection = taxa_filtering(readcounts_df_train)
    readcounts_df_train = readcounts_df_train.loc[:, selection]
    df_clr_train = centered_log_transform(readcounts_df_train)

    readcounts_df_test = taxa_aggregation(readcounts_df_test)
    readcounts_df_test = readcounts_df_test.loc[:, selection]
    df_clr_test = centered_log_transform(readcounts_df_test)

    df_train = pheno_df_train.join(df_clr_train)
    df_test = pheno_df_test.join(df_clr_test)
    selection = (df_train.columns != "Event") & (df_train.columns != "Event_time")
    covariates = df_train.columns[selection]

    X_train, X_test, y_train, y_test, test_sample_ids = prepare_train_test(
        df_train, df_test, covariates
    )
    return X_train, X_test, y_train, y_test, test_sample_ids


def standard_processing(
    pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test
):
    df_train = pheno_df_train.join(readcounts_df_train)
    df_test = pheno_df_test.join(readcounts_df_test)
    covariates = df_train.loc[
        :, (df_train.columns != "Event") & (df_train.columns != "Event_time")
    ].columns
    X_train, X_test, y_train, y_test, test_sample_ids = prepare_train_test(
        df_train, df_test, covariates
    )
    return X_train, X_test, y_train, y_test, test_sample_ids


def XGboost_formatting(X, y):
    event = y.Event
    time = y.Event_time

    data = xgb.DMatrix(X.to_numpy(), enable_categorical=True)

    # Associate ranged labels with the data matrix.
    y_lower_bound = np.array(time)
    y_upper_bound = np.array(time)
    y_upper_bound[event == 0] = np.inf
    data.set_float_info("label_lower_bound", y_lower_bound)
    data.set_float_info("label_upper_bound", y_upper_bound)

    return data
