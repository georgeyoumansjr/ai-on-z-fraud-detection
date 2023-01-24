
from django.http import HttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from .models import ModelF
from .forms import AddCsvFile
# from .preProcess import getOverView


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
import os
import json 
print(tf.__version__)

seq_length = 7 # Six past transactions followed by current transaction

from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.impute import SimpleImputer

class RunSimulation:

    def __init__(self, filePath):
        self.filePath = filePath
        if not self.filePath:
            self.filePath = os.path.join(settings.MEDIA_ROOT, "csvs\\card_transaction.v2.csv")
        print(self.filePath)
        self.df = pd.read_csv(self.filePath)
        # self.df = self.preArrange(self.df)
        # self.save_dir = os.path.join(settings.MEDIA_ROOT,'saved_models\\P\\ccf_220_keras_gru_static/1')

        # # Get first of each User-Card combination
        # first = self.df[['User','Card']].drop_duplicates()
        # f = np.array(first.index)

        # # Drop the first N transactions
        # drop_list = np.concatenate([np.arange(x,x + seq_length - 1) for x in f])
        # index_list = np.setdiff1d(self.df.index.values,drop_list)

        # # Split into 0.5 train, 0.3 validate, 0.2 test
        # tot_length = index_list.shape[0]
        # train_length = tot_length // 2
        # validate_length = (tot_length - train_length) * 3 // 5
        # test_length = tot_length - train_length - validate_length
        # print (tot_length,train_length,validate_length, test_length)

        # # Generate list of indices for train, validate, test
        # np.random.seed(1111)
        # train_indices = np.random.choice(index_list, train_length, replace=False)
        # tv_list = np.setdiff1d(index_list, train_indices)
        # validate_indices = np.random.choice(tv_list, validate_length, replace=False)
        # test_indices = np.setdiff1d(tv_list, validate_indices)
        # print (train_indices, validate_indices, test_indices)

        # self.create_test_sample(self.df, validate_indices[:100000]) # Uncomment this line to generate a test sample                    
        
    def getDf(self):
        return self.df

    def preArrange(self, df):
        self.df = df
        self.df['Merchant Name'] = self.df['Merchant Name'].astype(str)
        self.df['Merchant City'] = self.df['Merchant City'].astype(str)
        self.df.sort_values(by=['User','Card'], inplace=True)
        self.df.reset_index(inplace=True, drop=True)
        return self.df
    
    def create_test_sample(self,df, indices):
        print(indices)
        rows = indices.shape[0]
        index_array = np.zeros((rows, seq_length), dtype=int)
        for i in range(seq_length):
            index_array[:,i] = indices + 1 - seq_length + i
        uniques = np.unique(index_array.flatten())
        df.loc[uniques].to_csv(os.path.join(settings.MEDIA_ROOT, "testD/test_220_100k.csv"),index_label='Index')
        np.savetxt('test_220_100k.indices',indices.astype(int),fmt='%d')
    
    def timeEncoder(self,X):
        X_hm = X['Time'].str.split(':', expand=True)
        d = pd.to_datetime(dict(year=X['Year'],month=X['Month'],day=X['Day'],hour=X_hm[0],minute=X_hm[1]))
        d = d.values.astype(int)
        return pd.DataFrame(d)

    def amtEncoder(self,X):
        amt = X.apply(lambda x: x[1:]).astype(float).map(lambda amt: max(1,amt)).map(math.log)
        return pd.DataFrame(amt)

    def decimalEncoder(self,X,length=5):
        dnew = pd.DataFrame()
        for i in range(length):
            dnew[i] = np.mod(X,10) 
            X = np.floor_divide(X,10)
        return dnew

    def fraudEncoder(self,X):
        return np.where(X == 'Yes', 1, 0).astype(int)
    
    def generateMapper(self):
        mapper = DataFrameMapper([('Is Fraud?', FunctionTransformer(self.fraudEncoder)),
                          (['Merchant State'], [SimpleImputer(strategy='constant'), FunctionTransformer(np.ravel),
                                               LabelEncoder(), FunctionTransformer(self.decimalEncoder), OneHotEncoder()]),
                          (['Zip'], [SimpleImputer(strategy='constant'), FunctionTransformer(np.ravel),
                                     FunctionTransformer(self.decimalEncoder), OneHotEncoder()]),
                          ('Merchant Name', [LabelEncoder(), FunctionTransformer(self.decimalEncoder), OneHotEncoder()]),
                          ('Merchant City', [LabelEncoder(), FunctionTransformer(self.decimalEncoder), OneHotEncoder()]),
                          ('MCC', [LabelEncoder(), FunctionTransformer(self.decimalEncoder), OneHotEncoder()]),
                          (['Use Chip'], [SimpleImputer(strategy='constant'), LabelBinarizer()]),
                          (['Errors?'], [SimpleImputer(strategy='constant'), LabelBinarizer()]),
                          (['Year','Month','Day','Time'], [FunctionTransformer(self.timeEncoder), MinMaxScaler()]),
                          ('Amount', [FunctionTransformer(self.amtEncoder), MinMaxScaler()])
                         ], input_df=True, df_out=True)
        mapper.fit(self.df)

        joblib.dump(mapper, open(os.path.join(settings.MEDIA_ROOT,'fitted_mapper.pkl'),'wb'))








def overview(request):
    # path = os.path.join(settings.MEDIA_ROOT, "card_transaction.v2.csv")
    # print(path)
    print("FileName in Overview")
    id = request.session["id"]
    # print("Id is "+str(id))
    overD = ModelF.objects.get(id=id)
    # filePath = settings.MEDIA_URL + overD.csv_file
    # print(filePath)
    tdf = RunSimulation(overD.csv_file)
    img = os.path.abspath(os.path.join(settings.BASE_DIRV,"model.png"))
    # print(settings.BASE_DIRV)
    json_f = open(os.path.abspath(os.path.join(settings.BASE_DIRV,"metrics.json")),'r')
    fileR = json.load(json_f)
    
    # print(fileR)
    
    tdfH = tdf.getDf().head(20)
    fields = {k:str(v[0]) for k,v in pd.DataFrame(tdfH.dtypes).T.to_dict('list').items()}
    tdfH = tdfH.style.set_table_attributes('class="table table-info table-striped"')
    # print(fields)
    # print(tdf.info().to_html())
    mydict = {
        "df": tdfH.to_html(index=False),
        "metrics": fileR["metrics"],
        "fields": fields,
        "img": img
    }
    return render(request,'overview.html',context=mydict)
    # return HttpResponse(tdfH.to_html(index=False))

def uploadView(request):
    return 
from django.forms.models import model_to_dict

def addCsvView(request):
    if request.method == 'POST':
        # form = UserCreationForm(request.POST)
        form = AddCsvFile(request.POST,request.FILES)
        if form.is_valid():
            cFile,created = ModelF.objects.get_or_create(**form.cleaned_data)
            
            request.session["id"] = cFile.id
            print("Id is " + str(cFile.id))
            return redirect('overview')
    else:
        # form = UserCreationForm()
        form = AddCsvFile()
    return render(request,'uploadView.html',{'form': form})

