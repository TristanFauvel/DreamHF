# %%
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



def clinical_covariates_selection(X_train, y_train, clinical_covariates):     
    min_features_to_select = 1  # Minimum number of features to consider    
    model = CoxPH(0)
    cv = RepeatedKFold(n_splits = 10, n_repeats = 20)

    features = np.intersect1d(clinical_covariates, X_train.columns)
    other_features = np.setxor1d(clinical_covariates, X_train.columns)
    rfecv = RFECV(
        estimator=model.pipeline[1],
        step=1,
        cv=cv,
        min_features_to_select=min_features_to_select,
        n_jobs=-1,
        verbose = 0
    )
    rfecv.fit(model.pipeline[0].fit_transform(X_train.loc[:,features], y_train), y_train)
    features = features[rfecv.support_]
    output = np.union1d(features, other_features)
    return output



def relative_abundance(readcounts_df):
    total = readcounts_df.sum(axis=1)
    proportions_df = readcounts_df.divide(total, axis="rows")
    return proportions_df

def taxa_presence(readcounts_df):
    total = readcounts_df.sum(axis=1)
    df_proportions = readcounts_df.divide(total, axis="rows")
    presence = (df_proportions > 1e-5)
    return presence

def _pheno_processing_pipeline(df, training) -> pd.DataFrame:
    df = df.convert_dtypes()

    if "Event" in df:
        df.dropna(subset=["Event"], inplace=True)
        df = df.astype({"Event": "bool"})

    if "Event_time" in df:
        df.dropna(subset=["Event_time"], inplace=True)
        df = df.astype({"Event_time": "float64"})
          
    #df = df.astype({'Smoking': 'category', 'PrevalentCHD': 'category', 'BPTreatment': 'category', 'PrevalentDiabetes': 'category', 'PrevalentHFAIL': 'category',
                   # 'Sex': 'category', 'Event': 'category'})

    df.set_index("Unnamed: 0", inplace=True)

    df = df.rename_axis(index=None, columns=df.index.name)

    if training:
        artifacts = (df["Event_time"] < 0) & (df["Event"] == 1)
        df = df.loc[~artifacts, :]

    return df


def _readcounts_processing_pipeline(df) -> pd.DataFrame:
    df = df.transpose()
    df.columns = df.iloc[0]
    df = df.drop(labels=["Unnamed: 0"], axis=0)
    df = df.astype(np.int64)
    return df


def _remove_unique_columns(df_train, df_test) -> Tuple[pd.DataFrame, pd.DataFrame]:
    for col in df_test.columns:
        if len(df_test[col].unique()) == 1 and len(df_train[col].unique()) == 1:
            df_test.drop(col, inplace=True, axis=1)
            df_train.drop(col, inplace=True, axis=1)
    return df_train, df_test


def load_data(root, scoring = False):
    # Load data from files
    pheno_df_train = pd.read_csv(root + "/train/pheno_training.csv")
    pheno_df_train = _pheno_processing_pipeline(pheno_df_train, training=True)

    pheno_df_test = pd.read_csv(root + "/test/pheno_test.csv")
    pheno_df_test = _pheno_processing_pipeline(pheno_df_test, training=True)
    
    readcounts_df_train = pd.read_csv(root + "/train/readcounts_training.csv")
    readcounts_df_train = _readcounts_processing_pipeline(readcounts_df_train)
    
    readcounts_df_test = pd.read_csv(root + "/test/readcounts_test.csv")
    readcounts_df_test = _readcounts_processing_pipeline(readcounts_df_test)
    
    if scoring:
        pheno_df_scoring = pd.read_csv(root + "/scoring/pheno_scoring.csv")
        pheno_df_scoring = _pheno_processing_pipeline(
            pheno_df_scoring, training=False)

        readcounts_df_scoring = pd.read_csv(
            root + "/scoring/readcounts_scoring.csv")
        readcounts_df_scoring = _readcounts_processing_pipeline(
            readcounts_df_scoring)

        pheno_df_train = pd.concat([pheno_df_train, pheno_df_test])
        pheno_df_test = pheno_df_scoring

        readcounts_df_train = pd.concat([readcounts_df_train, readcounts_df_test])
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
    # To avoid the ValueError: feature_names must be string, and may not contain[, ] or < when using XGBoost
    #df.columns = df.columns.str.replace(
    #    r"[[]><]", "")
    invalid_characters = ['[',']','>','<']
    for c in invalid_characters:
        df.columns = [col.replace(c, '') for col in df.columns] 
    return df
    
def _prepare_train_test(df_train: pd.DataFrame, df_test: pd.DataFrame, covariates):
    # Left truncation : we remove all participants who experienced HF before entering the study.
    selection_train = df_train.loc[:, "Event_time"] >= -np.inf  # 0

    test_sample_ids = df_test.index
    train_sample_ids = df_train.index
    
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

    return X_train, X_test, y_train, y_test, test_sample_ids, train_sample_ids


def _check_data(df: pd.DataFrame) -> pd.DataFrame:
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


def _taxa_aggregation(readcounts_df, taxonomic_level="s__") -> pd.DataFrame:
    # Aggregate the species into genus
    readcounts_df.columns = [
        el.split(taxonomic_level)[0] for el in readcounts_df.columns
    ]
    readcounts_df = readcounts_df.groupby(readcounts_df.columns, axis=1).sum()
    return readcounts_df


def _taxa_filtering(readcounts_df):
    ## Select species-level taxonomic groups that were detected in >1% of the study participants at a within-sample relative abundance of >0.1%. 
    df_proportions  = relative_abundance(readcounts_df)
    selection = (df_proportions > 0.001).mean(axis=0) > 0.01
    readcounts_df = readcounts_df.loc[:, selection]

    # Median relative abundance of the selected genus
    relative_abundance = df_proportions.loc[:, selection].sum(axis=1)
    relative_abundance.median()

    return selection


def _centered_log_transform(readcounts_df) -> pd.DataFrame:
    ## Centered log transformation
    X_mr = multiplicative_replacement(readcounts_df)

    # CLR
    X_clr = clr(X_mr)

    df = pd.DataFrame(X_clr, columns=readcounts_df.columns, index=readcounts_df.index)
    return df


def Salosensaari_processing(
    pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test, clinical_covariates
):
    
    if "Event" in pheno_df_test:
        event_test = ["Event", "Event_time"]
    else:
        event_test = []
        
    pheno_df_train = pheno_df_train.loc[:, clinical_covariates + ["Event", "Event_time"]]
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
    selection = (df_train.columns != "Event") & (df_train.columns != "Event_time")
    covariates = df_train.columns[selection]

    X_train, X_test, y_train, y_test, test_sample_ids, train_sample_ids = _prepare_train_test(
        df_train, df_test, covariates
    )
    return X_train, X_test, y_train, y_test, test_sample_ids, train_sample_ids


def clr_processing(pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test, clinical_covariates, n_taxa):      
    
    adiv_train = diversity_metrics(
        readcounts_df_train, 'observed_otus').astype('float64')
    adiv_test = diversity_metrics(
        readcounts_df_test, 'observed_otus').astype('float64')

    
    if n_taxa > 0:
        taxa = taxa_selection(pheno_df_train, readcounts_df_train, n_taxa)
    else :
        taxa= None  
    
    
    #readcounts_df_train = taxa_presence(readcounts_df_train)
    #readcounts_df_test = taxa_presence(readcounts_df_test)
           
    if any(np.setdiff1d(clinical_covariates, CLINICAL_COVARIATES)):
        raise(ValueError('One of the clinical covariates is not in the prespecified list'))
    
    if "Event" in pheno_df_test:
        event_test = ["Event", "Event_time"]
    else:
        event_test = []

    pheno_df_train = pheno_df_train.loc[:, list(clinical_covariates)+
        ["Event", "Event_time"]]
    pheno_df_test = pheno_df_test.loc[:, list(clinical_covariates)+
        event_test]
    
    
    if taxa is None:
        df_train = pheno_df_train
        df_test = pheno_df_test
    else:
        df_clr_train = _centered_log_transform(readcounts_df_train)
        df_clr_test = _centered_log_transform(readcounts_df_test)
       
        df_train = pheno_df_train.join(df_clr_train.loc[:,taxa])
        df_test = pheno_df_test.join(df_clr_test.loc[:,taxa])  
    
    
    df_train['adiv'] = adiv_train
    df_test['adiv'] = adiv_test
        
    covariates = df_train.loc[
            :, (df_train.columns != "Event") & (df_train.columns != "Event_time")
        ].columns
    X_train, X_test, y_train, y_test, test_sample_ids, train_sample_ids = _prepare_train_test(
        df_train, df_test, covariates
    )
    
    ## Select the features using a coxPH model
    features = clinical_covariates_selection(X_train, y_train, clinical_covariates)
    X_train = X_train.loc[:, features]
    X_test = X_test.loc[:,features]
    
    return X_train, X_test, y_train, y_test, test_sample_ids, train_sample_ids


def standard_processing(pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test, clinical_covariates, taxa = None):      
    
    if any(np.setdiff1d(clinical_covariates, CLINICAL_COVARIATES)):
        raise(ValueError('One of the clinical covariates is not in the prespecified list'))
    
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
        df_train = pheno_df_train.join(readcounts_df_train.loc[:,taxa])
        df_test = pheno_df_test.join(readcounts_df_test.loc[:, taxa])
        
    covariates = df_train.loc[
            :, (df_train.columns != "Event") & (df_train.columns != "Event_time")
        ].columns
    X_train, X_test, y_train, y_test, test_sample_ids, train_sample_ids = _prepare_train_test(
        df_train, df_test, covariates
    )
    return X_train, X_test, y_train, y_test, test_sample_ids, train_sample_ids


def taxa_selection(pheno_df_train, readcounts_df_train, n_taxa = 200):
    
    if n_taxa<=0:
        raise ValueError('n_taxa must be >0')
    event_df = pheno_df_train['Event']
    threshold = 1e-5
    proportions_df=relative_abundance(readcounts_df_train)
    presence_df=proportions_df >= threshold

    kbest = SelectKBest(
        sklearn.feature_selection.mutual_info_classif, k=n_taxa)

    kbest.fit(presence_df, event_df)
    return kbest.get_feature_names_out()


def diversity_metrics(readcounts_df, metric):
    X= readcounts_df.to_numpy()
    ids = readcounts_df.index
    adiv= alpha_diversity(metric, X, ids).to_numpy().reshape(-1,1) 
    return adiv