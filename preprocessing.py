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
    
    #df  = df.loc[df.loc[:,'Event_time']>=0, :]   
    
    #X = df.loc[:,['Age', 'BodyMassIndex', 'Smoking=1', 'BPTreatment=1',
    #   'PrevalentDiabetes=1', 'PrevalentCHD=1', 'SystolicBP', 'NonHDLcholesterol', 'Sex=1']] 
    #y = df.loc[:,['Event', 'Event_time']]
    
    return df
    
def readcounts_processing_pipeline(df): 
    df = df.transpose()
    df.columns = df.iloc[0] 
    df =df.drop(labels = ['Unnamed: 0'], axis = 0)
    return df
