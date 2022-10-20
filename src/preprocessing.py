# %%
import pandas as pd
import numpy as np
import pandas as pd
from sksurv.column import encode_categorical
 

def pheno_processing_pipeline(df):
    df.dropna(inplace=True)
    df = df.convert_dtypes()
    df = df.astype({'Smoking':'category', 'PrevalentCHD':'category', 'BPTreatment':'category', 'PrevalentDiabetes':'category', 'PrevalentHFAIL':'category',
                                        'Event':'bool', 'Sex':'category'})


    df.set_index('Unnamed: 0',inplace=True)

    df = df.rename_axis(index=None, columns=df.index.name)

    df = encode_categorical(df, columns = ['Smoking', 'BPTreatment', 'PrevalentDiabetes', 'PrevalentCHD', 'PrevalentHFAIL', 'Sex'])
     
    # Remove column PrevalentHFAIL=1
    df.pop('PrevalentHFAIL=1') 
    
    
    artifacts = (df["Event_time"] < 0) & (df["Event"] == 1) 
    df = df.loc[~artifacts,:]
    #df  = df.loc[df.loc[:,'Event_time']>=0, :]   
    
    #X = df.loc[:,['Age', 'BodyMassIndex', 'Smoking=1', 'BPTreatment=1',
    #   'PrevalentDiabetes=1', 'PrevalentCHD=1', 'SystolicBP', 'NonHDLcholesterol', 'Sex=1']] 
    #y = df.loc[:,['Event', 'Event_time']]
    #df = df.rename(columns={df.columns[0]: 'ID'})
    
    return df
    
def readcounts_processing_pipeline(df): 
    df = df.transpose()
    df.columns = df.iloc[0] 
    df =df.drop(labels = ['Unnamed: 0'], axis = 0)
    return df


def prepare_train_test(df_train, df_test, covariates):
    
    # Left truncation : we remove all participants who experienced HF before entering the study.
    X_train = df_train.loc[df_train.loc[:,'Event_time']>=0, covariates]
    X_test = df_test.loc[df_test.loc[:,'Event_time']>=0,covariates]    
    y_train = df_train.loc[df_train.loc[:,'Event_time']>=0,['Event', 'Event_time']]
    y_test = df_test.loc[df_test.loc[:,'Event_time']>=0,['Event', 'Event_time']]
    
    """
    X_train = df_train.loc[:, covariates]
    X_test = df_test.loc[:,covariates]    
    y_train = df_train.loc[:,['Event', 'Event_time']]
    y_test = df_test.loc[:,['Event', 'Event_time']]
    """
    y_train =y_train.to_records(index = False)
    y_test =y_test.to_records(index = False)
    
    return X_train, X_test, y_train, y_test


def remove_unique_columns(readcounts_df_train, readcounts_df_test):
    for col in readcounts_df_test.columns:
        if len(readcounts_df_test[col].unique()) == 1 and len(readcounts_df_train[col].unique()) == 1:
            readcounts_df_test.drop(col,inplace=True,axis=1)
            readcounts_df_train.drop(col,inplace=True,axis=1)
    return readcounts_df_train, readcounts_df_test

 