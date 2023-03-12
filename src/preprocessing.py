# %%
import os
from typing import Tuple

import numpy as np
import pandas as pd
import sklearn
from skbio.diversity import alpha_diversity
from skbio.stats.composition import clr, multiplicative_replacement
from sklearn.feature_selection import RFECV, SelectKBest
from sklearn.model_selection import RepeatedKFold

from survival_models import CoxPH

CLINICAL_COVARIATES = ['Age', 'BodyMassIndex', 'Smoking', 'BPTreatment', 'PrevalentDiabetes',
                       'PrevalentCHD', 'PrevalentHFAIL', 'SystolicBP',
                       'NonHDLcholesterol', 'Sex']


def clinical_covariates_selection(x_train: pd.DataFrame, y_train: pd.Series, clinical_covariates: np.ndarray) -> np.ndarray:
    """
    Select the clinical covariates using Recursive Feature Elimination (RFE) with cross-validation.

    Args:
        x_train (pd.DataFrame): The training set.
        y_train (pd.Series): The target variable of the training set.
        clinical_covariates (np.ndarray): The array containing the clinical covariates to consider.

    Returns:
        np.ndarray: The selected clinical covariates.

    """
    min_features_to_select = 1  # Minimum number of features to consider
    model = CoxPH(0)
    cv = RepeatedKFold(n_splits=10, n_repeats=20)

    # Get the common features between clinical_covariates and x_train
    features = np.intersect1d(clinical_covariates, x_train.columns)

    # Get the features that are not in clinical_covariates
    other_features = np.setxor1d(clinical_covariates, x_train.columns)

    # Define the RFECV object
    rfecv = RFECV(
        estimator=model.pipeline[1],
        step=1,
        cv=cv,
        min_features_to_select=min_features_to_select,
        n_jobs=-1,
        verbose=0
    )

    # Fit the RFECV object to the training set
    rfecv.fit(model.pipeline[0].fit_transform(
        x_train.loc[:, features], y_train), y_train)

    # Get the selected features and add the other features to the output
    features = features[rfecv.support_]
    output = np.union1d(features, other_features)
    return output


def relative_abundance(readcounts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the relative abundance of each taxon in each sample.

    Args:
        readcounts_df (pd.DataFrame): The DataFrame containing the read counts.

    Returns:
        pd.DataFrame: The DataFrame containing the relative abundance of each taxon.

    """
    total = readcounts_df.sum(axis=1)
    proportions_df = readcounts_df.divide(total, axis="rows")
    return proportions_df


def taxa_presence(readcounts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine if each taxon is present or absent in each sample.

    Args:
        readcounts_df (pd.DataFrame): The DataFrame containing the read counts.

    Returns:
        pd.DataFrame: The DataFrame containing the presence (True) or absence (False) of each taxon.

    """
    total = readcounts_df.sum(axis=1)
    df_proportions = readcounts_df.divide(total, axis="rows")
    presence = (df_proportions > 1e-5)
    return presence


def _pheno_processing_pipeline(df: pd.DataFrame, training: bool) -> pd.DataFrame:
    """
    Process pheno dataframe by dropping rows with missing data, converting the data types of certain columns, 
    and setting the index.

    Args:
        df (pd.DataFrame): Dataframe to be processed
        training (bool): Whether or not the dataframe is from the training set

    Returns:
        pd.DataFrame: Processed dataframe
    """
    # Convert data types of columns
    df = df.convert_dtypes()

    # Drop rows with missing data for 'Event' and set type to boolean
    if "Event" in df:
        df.dropna(subset=["Event"], inplace=True)
        df = df.astype({"Event": "bool"})

    # Drop rows with missing data for 'Event_time' and set type to float64
    if "Event_time" in df:
        df.dropna(subset=["Event_time"], inplace=True)
        df = df.astype({"Event_time": "float64"})

    # Set index to column with label 'Unnamed: 0'
    df.set_index("Unnamed: 0", inplace=True)

    # Rename index and columns for clarity
    df = df.rename_axis(index=None, columns=df.index.name)

    # Remove rows with negative Event_time values in the training set (artifacts)
    if training:
        artifacts = (df["Event_time"] < 0) & (df["Event"] == 1)
        df = df.loc[~artifacts, :]

    return df


def _readcounts_processing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process readcounts dataframe by transposing, renaming columns, dropping labels, and converting data types.

    Args:
        df (pd.DataFrame): Dataframe to be processed

    Returns:
        pd.DataFrame: Processed dataframe
    """
    # Transpose dataframe
    df = df.transpose()

    # Rename columns to be first row of dataframe
    df.columns = df.iloc[0]

    # Drop row with label 'Unnamed: 0'
    df = df.drop(labels=["Unnamed: 0"], axis=0)

    # Convert data types of columns to int64
    df = df.astype(np.int64)

    return df


def _remove_unique_columns(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove columns in both training and testing sets with only one unique value.

    Args:
        df_train (pd.DataFrame): Training set
        df_test (pd.DataFrame): Testing set

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing modified training and testing sets
    """
    for col in df_test.columns:
        # If column has only one unique value in both training and testing sets, drop column
        if len(df_test[col].unique()) == 1 and len(df_train[col].unique()) == 1:
            df_test.drop(col, inplace=True, axis=1)
            df_train.drop(col, inplace=True, axis=1)
    return df_train, df_test


def load_data(root, scoring=False):
    """
    Load data from CSV files.

    Args:
        root (str): Path to directory containing the CSV files.
        scoring (bool, optional): If True, return data for scoring. Defaults to False.

    Returns:
        tuple: Tuple containing four DataFrames in the following order: 
            pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test
    """

    # Load data from files
    pheno_df_train = pd.read_csv(root + "/train/pheno_training.csv")
    pheno_df_train = _pheno_processing_pipeline(pheno_df_train, training=True)

    readcounts_df_train = pd.read_csv(root + "/train/readcounts_training.csv")
    readcounts_df_train = _readcounts_processing_pipeline(readcounts_df_train)

    if os.path.exists(root + '/test'):
        testing = True
    else:
        testing = False

    if testing:
        pheno_df_test = pd.read_csv(root + "/test/pheno_test.csv")
        pheno_df_test = _pheno_processing_pipeline(
            pheno_df_test, training=True)
        readcounts_df_test = pd.read_csv(root + "/test/readcounts_test.csv")
        readcounts_df_test = _readcounts_processing_pipeline(
            readcounts_df_test)

    if scoring:
        pheno_df_scoring = pd.read_csv(root + "/scoring/pheno_scoring.csv")
        pheno_df_scoring = _pheno_processing_pipeline(
            pheno_df_scoring, training=False)

        readcounts_df_scoring = pd.read_csv(
            root + "/scoring/readcounts_scoring.csv")
        readcounts_df_scoring = _readcounts_processing_pipeline(
            readcounts_df_scoring)

        if testing:
            pheno_df_train = pd.concat([pheno_df_train, pheno_df_test])
            readcounts_df_train = pd.concat(
                [readcounts_df_train, readcounts_df_test])

        pheno_df_test = pheno_df_scoring
        readcounts_df_test = readcounts_df_scoring

    # Remove unique columns from the readcounts data
    readcounts_df_train, readcounts_df_test = _remove_unique_columns(
        readcounts_df_train, readcounts_df_test
    )

    # Check the correspondence between the two tables
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

    readcounts_df_train = remove_invalid_characters(readcounts_df_train)
    readcounts_df_test = remove_invalid_characters(readcounts_df_test)

    return pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test


def remove_invalid_characters(df):
    """
    Removes invalid characters from DataFrame column names to avoid errors in XGBoost.
    
    Args:
        df (pandas.DataFrame): DataFrame with column names to be cleaned.
    
    Returns:
        pandas.DataFrame: DataFrame with cleaned column names.
    """
    invalid_characters = ['[', ']', '>', '<']
    for character in invalid_characters:
        df.columns = [col.replace(character, '') for col in df.columns]
    return df


def _prepare_train_test(df_train: pd.DataFrame, df_test: pd.DataFrame, covariates):
    """
    Prepares the training and testing datasets for the model by selecting relevant columns and removing invalid values.
    
    Args:
        df_train (pandas.DataFrame): DataFrame containing the training data.
        df_test (pandas.DataFrame): DataFrame containing the testing data.
        covariates (list): List of column names to be used as features for the model.
    
    Returns:
        tuple: A tuple containing the following:
            pandas.DataFrame: DataFrame containing the training data.
            pandas.DataFrame: DataFrame containing the testing data.
            numpy.ndarray: Array containing the training labels.
            numpy.ndarray: Array containing the testing labels.
            pandas.Index: Index of the samples in the testing data.
            pandas.Index: Index of the samples in the training data.
    """
    # Left truncation: we remove all participants who experienced HF before entering the study.
    selection_train = df_train.loc[:, "Event_time"] >= -np.inf  # 0

    test_sample_ids = df_test.index
    train_sample_ids = df_train.index

    # Make sure that the features do not contain Event or Event_time
    if "Event" in covariates or "Event_time" in covariates:
        Exception(
            "Event or Event_time are included in covariates, please remove them.")

    x_train = df_train.loc[selection_train, covariates]
    x_test = df_test.loc[:, covariates]
    y_train = df_train.loc[selection_train, ["Event", "Event_time"]]
    y_train = y_train.to_records(index=False)

    if "Event" in df_test:
        y_test = df_test.loc[:, ["Event", "Event_time"]]
        y_test = y_test.to_records(index=False)
    else:
        y_test = None

    return x_train, x_test, y_train, y_test, test_sample_ids, train_sample_ids


def _check_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks the input data for NaN values and deletes any rows with missing values.
    
    Args:
        df (pandas.DataFrame): DataFrame to be checked.
    
    Returns:
        pandas.DataFrame: DataFrame with missing values removed.
    """
    # Check that the input data does not contain NaN
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


def _taxa_filtering(readcounts_df):
    """
    Select species-level taxonomic groups that were detected in >1% of the study participants at a within-sample relative 
    abundance of >0.1%.
    
    Args:
    - readcounts_df: pandas.DataFrame. The input dataframe containing the read counts data.
    
    Returns:
    - selection: pandas.Series. A boolean series that indicates which species-level taxonomic groups are selected.
    """
    df_proportions = relative_abundance(readcounts_df)
    selection = (df_proportions > 0.001).mean(axis=0) > 0.01
    readcounts_df = readcounts_df.loc[:, selection]

    # Median relative abundance of the selected genus
    relative_abundance = df_proportions.loc[:, selection].sum(axis=1)
    relative_abundance.median()

    return selection


def _centered_log_transform(readcounts_df) -> pd.DataFrame:
    """
    Performs centered log transformation on read count data.
    
    Args:
    - readcounts_df: pandas.DataFrame. The input dataframe containing the read counts data.
    
    Returns:
    - df: pandas.DataFrame. The transformed dataframe.
    """
    # Centered log transformation
    x_mr = multiplicative_replacement(readcounts_df)

    # CLR
    x_clr = clr(x_mr)

    df = pd.DataFrame(x_clr, columns=readcounts_df.columns,
                      index=readcounts_df.index)
    return df


def Salosensaari_processing(
    pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test, clinical_covariates
):
    """
    Preprocesses data for Salosensaari et al. method. Performs taxa aggregation, filtering, and centered log transformation 
    on read count data, and prepares train and test data for the Cox proportional hazards model.
    
    Args:
    - pheno_df_train: pandas.DataFrame. The input dataframe containing the clinical data for the training set.
    - pheno_df_test: pandas.DataFrame. The input dataframe containing the clinical data for the test set.
    - readcounts_df_train: pandas.DataFrame. The input dataframe containing the read counts data for the training set.
    - readcounts_df_test: pandas.DataFrame. The input dataframe containing the read counts data for the test set.
    - clinical_covariates: list of str. The list of clinical covariates to be used in the analysis.
    
    Returns:
    - x_train: pandas.DataFrame. The training data.
    - x_test: pandas.DataFrame. The test data.
    - y_train: pandas.Series. The training data labels.
    - y_test: pandas.Series. The test data labels.
    - test_sample_ids: list of str. The sample IDs for the test set.
    - train_sample_ids: list of str. The sample IDs for the training set.
    """

    if "Event" in pheno_df_test:
        event_test = ["Event", "Event_time"]
    else:
        event_test = []

    pheno_df_train = pheno_df_train.loc[:,
                                        clinical_covariates + ["Event", "Event_time"]]
    pheno_df_test = pheno_df_test.loc[:, clinical_covariates + event_test]

    readcounts_df_train = _taxa_aggregation(readcounts_df_train)
    selection = _taxa_filtering(readcounts_df_train)
    readcounts_df_train = readcounts_df_train.loc[:, selection]
    df_clr_train = _centered_log_transform(readcounts_df_train)

    readcounts_df_test = _taxa_aggregation(readcounts_df_test)
    readcounts_df_test = readcounts_df_test.loc[:, selection]
    df_clr_test = _centered_log_transform(readcounts_df_test)

    df_train = pheno_df_train.join(df_clr_train)
    df_test = pheno_df_test.join(df_clr_test)
    selection = (df_train.columns != "Event") & (
        df_train.columns != "Event_time")
    covariates = df_train.columns[selection]

    x_train, x_test, y_train, y_test, test_sample_ids, train_sample_ids = _prepare_train_test(
        df_train, df_test, covariates
    )
    return x_train, x_test, y_train, y_test, test_sample_ids, train_sample_ids


def clr_processing(pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test, clinical_covariates, n_taxa):
    """
    Performs data processing and feature selection for Cox proportional hazards modeling using centered log-ratio transformation.

    Parameters:
    - pheno_df_train (pandas.DataFrame): phenotype dataframe for training set
    - pheno_df_test (pandas.DataFrame): phenotype dataframe for testing set
    - readcounts_df_train (pandas.DataFrame): read count dataframe for training set
    - readcounts_df_test (pandas.DataFrame): read count dataframe for testing set
    - clinical_covariates (list): list of clinical covariates to include in the model
    - n_taxa (int): number of taxa to include in the model; if 0, no taxa are included
    
    Returns:
    - x_train (pandas.DataFrame): feature matrix for training set
    - x_test (pandas.DataFrame): feature matrix for testing set
    - y_train (pandas.Series): event (1) or censoring (0) for training set
    - y_test (pandas.Series): event (1) or censoring (0) for testing set
    - test_sample_ids (list): sample IDs for testing set
    - train_sample_ids (list): sample IDs for training set
    """

    # Calculate alpha diversity for training and testing set
    adiv_train = diversity_metrics(
        readcounts_df_train, 'observed_otus').astype('float64')
    adiv_test = diversity_metrics(
        readcounts_df_test, 'observed_otus').astype('float64')

    # If n_taxa is specified, perform taxa selection using centered log-ratio transformation
    if n_taxa > 0:
        taxa = taxa_selection(pheno_df_train, readcounts_df_train, n_taxa)
    else:
        taxa = None

    # Check if all clinical covariates are valid
    if any(np.setdiff1d(clinical_covariates, CLINICAL_COVARIATES)):
        raise(ValueError('One of the clinical covariates is not in the prespecified list'))

    # Select relevant columns from phenotype dataframes
    if "Event" in pheno_df_test:
        event_test = ["Event", "Event_time"]
    else:
        event_test = []

    pheno_df_train = pheno_df_train.loc[:, list(clinical_covariates) +
                                        ["Event", "Event_time"]]
    pheno_df_test = pheno_df_test.loc[:, list(clinical_covariates) +
                                      event_test]

    if taxa is None:
        df_train = pheno_df_train
        df_test = pheno_df_test
    else:
        df_clr_train = _centered_log_transform(readcounts_df_train)
        df_clr_test = _centered_log_transform(readcounts_df_test)

        df_train = pheno_df_train.join(df_clr_train.loc[:, taxa])
        df_test = pheno_df_test.join(df_clr_test.loc[:, taxa])

    df_train['adiv'] = adiv_train
    df_test['adiv'] = adiv_test

    covariates = df_train.loc[
        :, (df_train.columns != "Event") & (df_train.columns != "Event_time")
    ].columns
    x_train, x_test, y_train, y_test, test_sample_ids, train_sample_ids = _prepare_train_test(
        df_train, df_test, covariates
    )

    # Select the features using a coxPH model
    features = clinical_covariates_selection(
        x_train, y_train, clinical_covariates)
    x_train = x_train.loc[:, features]
    x_test = x_test.loc[:, features]

    return x_train, x_test, y_train, y_test, test_sample_ids, train_sample_ids


def standard_processing(pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test, clinical_covariates, taxa=None):
    """
    Process and prepare the input data for modeling.

    Parameters:
    -----------
    pheno_df_train : pandas.DataFrame
        DataFrame of clinical phenotype data for the training set.
    pheno_df_test : pandas.DataFrame
        DataFrame of clinical phenotype data for the test set.
    readcounts_df_train : pandas.DataFrame
        DataFrame of read count data for the training set.
    readcounts_df_test : pandas.DataFrame
        DataFrame of read count data for the test set.
    clinical_covariates : list
        List of clinical covariates to be included in the modeling.
    taxa : list or None, default=None
        List of taxa to be included in the modeling, or None if only clinical covariates should be used.

    Returns:
    --------
    x_train : pandas.DataFrame
        DataFrame of training set predictors.
    x_test : pandas.DataFrame
        DataFrame of test set predictors.
    y_train : pandas.DataFrame
        DataFrame of training set outcomes.
    y_test : pandas.DataFrame
        DataFrame of test set outcomes.
    test_sample_ids : list
        List of sample IDs for the test set.
    train_sample_ids : list
        List of sample IDs for the training set.
    """
    # Check that all specified clinical covariates are valid
    if any(np.setdiff1d(clinical_covariates, CLINICAL_COVARIATES)):
        raise(ValueError('One of the clinical covariates is not in the prespecified list'))

    # Check if the test set includes event data
    if "Event" in pheno_df_test:
        event_test = ["Event", "Event_time"]
    else:
        event_test = []

    # Select relevant columns from the input DataFrames
    pheno_df_train = pheno_df_train.loc[:, list(
        clinical_covariates) + ["Event", "Event_time"]]
    pheno_df_test = pheno_df_test.loc[:, list(
        clinical_covariates) + event_test]

    # Merge read count data with the clinical phenotype data, if specified
    if taxa is None:
        df_train = pheno_df_train
        df_test = pheno_df_test
    else:
        df_train = pheno_df_train.join(readcounts_df_train.loc[:, taxa])
        df_test = pheno_df_test.join(readcounts_df_test.loc[:, taxa])

    # Extract covariate columns for modeling
    covariates = df_train.loc[:, (df_train.columns != "Event") & (
        df_train.columns != "Event_time")].columns

    # Split the data into training and test sets
    x_train, x_test, y_train, y_test, test_sample_ids, train_sample_ids = _prepare_train_test(
        df_train, df_test, covariates)

    return x_train, x_test, y_train, y_test, test_sample_ids, train_sample_ids


def taxa_selection(pheno_df_train, readcounts_df_train, n_taxa=200):
    """
    Performs taxa selection based on mutual information between presence of taxa and the event.

    Parameters:
    - pheno_df_train: pandas DataFrame with clinical covariates and event information for training set.
    - readcounts_df_train: pandas DataFrame with OTU read counts for training set.
    - n_taxa: integer representing the number of taxa to be selected based on mutual information. Default: 200

    Returns:
    - list of feature names with length n_taxa representing selected taxa.
    """
    if n_taxa <= 0:
        raise ValueError('n_taxa must be >0')
    event_df = pheno_df_train['Event']
    threshold = 1e-5
    proportions_df = relative_abundance(readcounts_df_train)
    presence_df = proportions_df >= threshold

    kbest = SelectKBest(
        sklearn.feature_selection.mutual_info_classif, k=n_taxa)

    kbest.fit(presence_df, event_df)
    return kbest.get_feature_names_out()


def diversity_metrics(readcounts_df, metric):
    """
    Calculates alpha diversity metrics based on the OTU read counts.

    Parameters:
    - readcounts_df: pandas DataFrame with OTU read counts.
    - metric: string representing the alpha diversity metric to calculate.

    Returns:
    - numpy array with the calculated alpha diversity metric values.
    """
    X = readcounts_df.to_numpy()
    ids = readcounts_df.index
    adiv = alpha_diversity(metric, X, ids).to_numpy().reshape(-1, 1)
    return adiv
