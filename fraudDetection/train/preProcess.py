import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import math
import os
import joblib
import pydot
import sys
from django.conf import settings

print(tf.__version__)

seq_length = 7 # Six past transactions followed by current transaction

def getOverview():
    tdf = pd.read_csv("../media/csvs/card_transaction.v2.csv")
    tdf['Merchant Name'] = tdf['Merchant Name'].astype(str)
    tdf['Merchant City'] = tdf['Merchant City'].astype(str)
    tdf.sort_values(by=['User','Card'], inplace=True)
    tdf.reset_index(inplace=True, drop=True)
    return tdf.info()

"""
# Get first of each User-Card combination
first = tdf[['User','Card']].drop_duplicates()
f = np.array(first.index)
print(f)


# **********************************************
# Drop the first N transactions
drop_list = np.concatenate([np.arange(x,x + seq_length - 1) for x in f])
print(drop_list)
index_list = np.setdiff1d(tdf.index.values,drop_list)
print(index_list)

# *********************************************
# Split into 0.5 train, 0.3 validate, 0.2 test
tot_length = index_list.shape[0]
train_length = tot_length // 2
validate_length = (tot_length - train_length) * 3 // 5
test_length = tot_length - train_length - validate_length
print (tot_length,train_length,validate_length, test_length)

# ************************************************

# Generate list of indices for train, validate, test
np.random.seed(1111)
train_indices = np.random.choice(index_list, train_length, replace=False)
print(train_indices)
tv_list = np.setdiff1d(index_list, train_indices)
print(tv_list)
validate_indices = np.random.choice(tv_list, validate_length, replace=False)
test_indices = np.setdiff1d(tv_list, validate_indices)
print(test_indices)
print (train_indices, validate_indices, test_indices)

# ************************************

def create_test_sample(df, indices):
    print(indices)
    rows = indices.shape[0]
    index_array = np.zeros((rows, seq_length), dtype=np.int)
    for i in range(seq_length):
        index_array[:,i] = indices + 1 - seq_length + i
    uniques = np.unique(index_array.flatten())
    df.loc[uniques].to_csv('test_220_100k.csv',index_label='Index')
    np.savetxt('test_220_100k.indices',indices.astype(int),fmt='%d')

create_test_sample(tdf, validate_indices[:100000]) # Uncomment this line to generate a test sample

# ****************************
def timeEncoder(X):
    X_hm = X['Time'].str.split(':', expand=True)
    d = pd.to_datetime(dict(year=X['Year'],month=X['Month'],day=X['Day'],hour=X_hm[0],minute=X_hm[1])).astype('int64')
    return pd.DataFrame(d)

def amtEncoder(X):
    amt = X.apply(lambda x: x[1:]).astype(float).map(lambda amt: max(1,amt)).map(math.log)
    return pd.DataFrame(amt)

def decimalEncoder(X,length=5):
    dnew = pd.DataFrame()
    for i in range(length):
        dnew[i] = np.mod(X,10) 
        X = np.floor_divide(X,10)
    return dnew

def fraudEncoder(X):
    return np.where(X == 'Yes', 1, 0).astype(int)

# *******************************************

save_dir = 'saved_models'

from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.impute import SimpleImputer

mapper = DataFrameMapper([('Is Fraud?', FunctionTransformer(fraudEncoder)),
                          (['Merchant State'], [SimpleImputer(strategy='constant'), FunctionTransformer(np.ravel),
                                               LabelEncoder(), FunctionTransformer(decimalEncoder), OneHotEncoder()]),
                          (['Zip'], [SimpleImputer(strategy='constant'), FunctionTransformer(np.ravel),
                                     FunctionTransformer(decimalEncoder), OneHotEncoder()]),
                          ('Merchant Name', [LabelEncoder(), FunctionTransformer(decimalEncoder), OneHotEncoder()]),
                          ('Merchant City', [LabelEncoder(), FunctionTransformer(decimalEncoder), OneHotEncoder()]),
                          ('MCC', [LabelEncoder(), FunctionTransformer(decimalEncoder), OneHotEncoder()]),
                          (['Use Chip'], [SimpleImputer(strategy='constant'), LabelBinarizer()]),
                          (['Errors?'], [SimpleImputer(strategy='constant'), LabelBinarizer()]),
                          (['Year','Month','Day','Time'], [FunctionTransformer(timeEncoder), MinMaxScaler()]),
                          ('Amount', [FunctionTransformer(amtEncoder), MinMaxScaler()])
                         ], input_df=True, df_out=True)
mapper.fit(tdf)

joblib.dump(mapper, open('fitted_mapper.pkl','wb'))

# ***************************
mapper = joblib.load(open('fitted_mapper.pkl','rb'))



class PreProcess:

    def __init__(self,fileName):
        self.df = pd.read_csv(settings.MEDIA_ROOT + fileName)
        self.df['Merchant Name'] = self.df['Merchant Name'].astype(str)
        self.df['Merchant City'] = self.df['Merchant City'].astype(str)
        self.df.sort_values(by=['User','Card'], inplace=True)
        self.df.reset_index(inplace=True, drop=True)

"""



