
from django.http import HttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from .models import ModelF
from .forms import AddCsvFile
# from .preProcess import getOverView


import tensorflow as tf
from tensorflow import keras

import pandas as pd
import math
import os
import joblib
import pydot
import numpy as np
import sys
from django.conf import settings
import os
import json 
print(tf.__version__)

seq_length = 7 # Six past transactions followed by current transaction
save_dir = 'saved_models\\P\\ccf_220_keras_gru_static\\1'

from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.impute import SimpleImputer


class TP(tf.keras.metrics.TruePositives):
    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true[-1,:,:], y_pred[-1,:,:], sample_weight)

class FP(tf.keras.metrics.FalsePositives):
    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true[-1,:,:], y_pred[-1,:,:], sample_weight)

class FN(tf.keras.metrics.FalseNegatives):
    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true[-1,:,:], y_pred[-1,:,:], sample_weight)

class TN(tf.keras.metrics.TrueNegatives):
    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true[-1,:,:], y_pred[-1,:,:], sample_weight)
metrics=['accuracy', 
    TP(name='TP'),
    FP(name='FP'),
    FN(name='FN'),
    TN(name='TN'),
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.TrueNegatives(name='tn')
   ]

class Preprocess:

    def __init__(self,filePath):
        self.filePath = filePath

        self.tdf = pd.read_csv(os.path.join(settings.BASE_DIRV,'Data/card_transaction.v1.csv'))
        self.tdf['Merchant Name'] = self.tdf['Merchant Name'].astype(str)
        self.tdf.sort_values(by=['User','Card'], inplace=True)
        self.tdf.reset_index(inplace=True, drop=True)
        print (self.tdf.info())

        # Get first of each User-Card combination
        first = self.tdf[['User','Card']].drop_duplicates()
        f = np.array(first.index)

        # Drop the first N transactions
        drop_list = np.concatenate([np.arange(x,x + seq_length - 1) for x in f])
        index_list = np.setdiff1d(self.tdf.index.values,drop_list)

        # Split into 0.5 train, 0.3 validate, 0.2 test
        tot_length = index_list.shape[0]
        train_length = tot_length // 2
        validate_length = (tot_length - train_length) * 3 // 5
        test_length = tot_length - train_length - validate_length
        print (tot_length,train_length,validate_length, test_length)

        # Generate list of indices for train, validate, test
        np.random.seed(1111)
        self.train_indices = np.random.choice(index_list, train_length, replace=False)
        tv_list = np.setdiff1d(index_list, self.train_indices)
        validate_indices = np.random.choice(tv_list, validate_length, replace=False)
        test_indices = np.setdiff1d(tv_list, validate_indices)
        print (self.train_indices, validate_indices, test_indices)
    
        # self.create_test_sample(self.tdf, validate_indices[:100000])

    def getDf(self):
        return self.tdf
    
    def getTrainIndices(self):
        return self.train_indices
        

    def create_test_sample(self,df, indices):
        print(indices)
        rows = indices.shape[0]
        index_array = np.zeros((rows, seq_length), dtype=np.int)
        for i in range(seq_length):
            index_array[:,i] = indices + 1 - seq_length + i
        uniques = np.unique(index_array.flatten())
        df.loc[uniques].to_csv('test_220_100k.csv',index_label='Index')
        np.savetxt('test_220_100k.indices',indices.astype(int),fmt='%d')










seq_length = 7
# import numpy as np
# def gen_test_batch(df, mapper, indices, batch_size):
#     mapper = joblib.load(open(os.path.join(settings.BASE_DIRV,'fitted_mapper.pkl'),'rb'))
#     rows = indices.shape[0]
#     index_array = np.zeros((rows, seq_length), dtype=int)
#     for i in range(seq_length):
#         index_array[:,i] = indices + 1 - seq_length + i
#     count = 0
#     while (count + batch_size <= rows):        
#         arr = index_array[count:count+batch_size].flatten()
#         print(arr)
#         full_df = mapper.transform(df.loc[arr])
#         data = full_df.drop(['Is Fraud?'],axis=1).to_numpy().reshape(batch_size, seq_length, -1)
#         targets = full_df['Is Fraud?'].to_numpy().reshape(batch_size, seq_length, 1)
#         count += batch_size
#         data_t = np.transpose(data, axes=(1,0,2))
#         targets_t = np.transpose(targets, axes=(1,0,2))
#         yield data_t, targets_t







def timeEncoder(X):
    X_hm = X['Time'].str.split(':', expand=True)
    d = pd.to_datetime(dict(year=X['Year'],month=X['Month'],day=X['Day'],hour=X_hm[0],minute=X_hm[1]))
    d = d.values.astype(int)
    print(d.dtype)
    d = d.astype(int)
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

def gen_training_batch(df, mapper, index_list, batch_size):
    np.random.seed(98765)
    train_df = df.loc[index_list]
    non_fraud_indices = train_df[train_df['Is Fraud?'] == 'No'].index.values
    fraud_indices = train_df[train_df['Is Fraud?'] == 'Yes'].index.values
    fsize = fraud_indices.shape[0]
    while True:
        indices = np.concatenate((fraud_indices,np.random.choice(non_fraud_indices,fsize,replace=False)))
        np.random.shuffle(indices)
        rows = indices.shape[0]
        index_array = np.zeros((rows, seq_length), dtype=np.int)
        for i in range(seq_length):
            index_array[:,i] = indices + 1 - seq_length + i
        full_df = mapper.transform(df.loc[index_array.flatten()])
        target_buffer = full_df['Is Fraud?'].to_numpy().reshape(rows, seq_length, 1)
        data_buffer = full_df.drop(['Is Fraud?'],axis=1).to_numpy().reshape(rows, seq_length, -1)

        batch_ptr = 0
        while (batch_ptr + batch_size) <= rows:
            data = data_buffer[batch_ptr:batch_ptr+batch_size]
            targets = target_buffer[batch_ptr:batch_ptr+batch_size]
            batch_ptr += batch_size
            data_t = np.transpose(data, axes=(1,0,2))
            targets_t = np.transpose(targets, axes=(1,0,2))
            yield data_t,targets_t

def print_trainable_parameters():
    total = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        parameters = 1
        for dim in shape:
            parameters *= dim
        total += parameters
        print (variable, shape, parameters)
    print(total)

def f1(conf):
    precision = float(conf[1][1]) / (conf[1][1]+conf[0][1])
    recall = float(conf[1][1]) / (conf[1][1]+conf[1][0])
    return 2 * precision * recall / (precision + recall)


def preprocess(request):

    # tdf = Preprocess("lol")
    # train_indices = tdf.getTrainIndices()
    # tdf = tdf.getDf()

    # print("Train",train_indices)
    # train(tdf,train_indices)
    data =  test()
    print(data)
    
    # mapper = DataFrameMapper([('Is Fraud?', FunctionTransformer(fraudEncoder)),
    #                       (['Merchant State'], [SimpleImputer(strategy='constant'), FunctionTransformer(np.ravel),
    #                                            LabelEncoder(), FunctionTransformer(decimalEncoder), OneHotEncoder()]),
    #                       (['Zip'], [SimpleImputer(strategy='constant'), FunctionTransformer(np.ravel),
    #                                  FunctionTransformer(decimalEncoder), OneHotEncoder()]),
    #                       ('Merchant Name', [LabelEncoder(), FunctionTransformer(decimalEncoder), OneHotEncoder()]),
    #                       ('Merchant City', [LabelEncoder(), FunctionTransformer(decimalEncoder), OneHotEncoder()]),
    #                       ('MCC', [LabelEncoder(), FunctionTransformer(decimalEncoder), OneHotEncoder()]),
    #                       (['Use Chip'], [SimpleImputer(strategy='constant'), LabelBinarizer()]),
    #                       (['Errors?'], [SimpleImputer(strategy='constant'), LabelBinarizer()]),
    #                       (['Year','Month','Day','Time'], [FunctionTransformer(timeEncoder), MinMaxScaler()]),
    #                       ('Amount', [FunctionTransformer(amtEncoder), MinMaxScaler()])
    #                      ], input_df=True, df_out=True)
    # mapper.fit(tdf)
    # joblib.dump(mapper, open(os.path.join(settings.BASE_DIR,'fitted_mapper.pkl'),'wb'))

    print("Second part complete")


    return HttpResponse("Suceh")




def train(tdf,train_indices):

    mapper = joblib.load(open(os.path.join(settings.BASE_DIR,'fitted_mapper.pkl'),'rb'))
    mapped_sample = mapper.transform(tdf[:100])
    mapped_size = mapped_sample.shape[-1]
    print(mapped_size)

    units = [200,200]
    input_size = mapped_size - 1
    output_size = 1

    batch_size = 16
    tf_input = ([batch_size, input_size])

    gru_model = tf.keras.models.Sequential([
        tf.keras.layers.GRU(units[0], input_shape=tf_input, batch_size=7, time_major=True, return_sequences=True),
        tf.keras.layers.GRU(units[1], return_sequences=True, time_major=True),
        tf.keras.layers.Dense(output_size, activation='sigmoid')
    ])

    gru_model.summary()
    # tf.keras.utils.plot_model(gru_model, 'model.png', show_shapes=True)
    gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)

    steps_per_epoch = 50000
    checkpoint_dir = os.path.join(settings.BASE_DIR, "checkpoints\\ccf_220_keras_gru_static\\")
    filepath = checkpoint_dir + "iter-{epoch:02d}/model.ckpt"
    batch_size = 16

    print ("Learning...")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, save_weights_only=True, verbose=1)
    train_generate = gen_training_batch(tdf,mapper,train_indices,batch_size)
    gru_model.fit(train_generate, epochs=1, steps_per_epoch=steps_per_epoch, verbose=1, callbacks=[cp_callback])

    gru_model.save_weights(os.path.join(settings.BASE_DIR,save_dir,"wts"))
    gru_model.save(save_dir)


def gen_test_batch(df, mapper, indices, batch_size):
    rows = indices.shape[0]
    index_array = np.zeros((rows, seq_length), dtype=np.int)
    for i in range(seq_length):
        index_array[:,i] = indices + 1 - seq_length + i
    count = 0
    while (count + batch_size <= rows):        
        full_df = mapper.transform(df.loc[index_array[count:count+batch_size].flatten()])
        data = full_df.drop(['Is Fraud?'],axis=1).to_numpy().reshape(batch_size, seq_length, -1)
        targets = full_df['Is Fraud?'].to_numpy().reshape(batch_size, seq_length, 1)
        count += batch_size
        data_t = np.transpose(data, axes=(1,0,2))
        targets_t = np.transpose(targets, axes=(1,0,2))
        yield data_t, targets_t

def test():
    batch_size = 2000

    input_size=220
    output_size=1
    units=[200,200]

    tf_input = ([batch_size, input_size])
    mapper = joblib.load(open(os.path.join(settings.BASE_DIR,'fitted_mapper.pkl'),'rb'))

    new_model = tf.keras.models.Sequential([
        tf.keras.layers.GRU(units[0], input_shape=tf_input, batch_size=7, time_major=True, return_sequences=True),
        tf.keras.layers.GRU(units[1], return_sequences=True, time_major=True),
        tf.keras.layers.Dense(output_size, activation='sigmoid')
    ])
    new_model.load_weights(os.path.join(settings.BASE_DIR,save_dir,"wts"))
    new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    ddf = pd.read_csv(os.path.join(settings.BASE_DIR,'test_220_100k.csv'), dtype={"Merchant Name":"str"}, index_col='Index')
    indices = np.loadtxt(os.path.join(settings.BASE_DIR,'test_220_100k.indices'))

    batch_size = 2000

    print("\nQuick test")
    test_generate = gen_test_batch(ddf,mapper,indices,batch_size)
    evaluatedR = new_model.evaluate(test_generate, verbose=0)
    print(evaluatedR)

    # print("\nFull test")
    # test_generate = gen_test_batch(tdf,mapper,test_indices,batch_size)
    # newEval = new_model.evaluate(test_generate, verbose=1)
    # print(newEval)
    return evaluatedR

def overview(request):

    
    print("FileName in Overview")
    id = request.session["id"]
    overD = ModelF.objects.get(id=id)

   
    # tdf = RunSimulation(overD.csv_file)
   
    # evaluatedR = testEvaluation(tdf.getDf())
    # print(evaluatedR)
    img = os.path.abspath(os.path.join(settings.BASE_DIRV,"model.png"))
    
    json_f = open(os.path.abspath(os.path.join(settings.BASE_DIRV,"metrics.json")),'r')
    fileR = json.load(json_f)
    
    # evaluatedR = testEvaluation(overD.csv_file)
    # tdf = Preprocess("a")

    

    
    tdfH = tdf.head(20)
    fields = {k:str(v[0]) for k,v in pd.DataFrame(tdfH.dtypes).T.to_dict('list').items()}
    tdfH = tdfH.style.set_table_attributes('class="table table-info table-striped"')
    
    mydict = {
        "df": tdfH.to_html(index=False),
        "metrics": fileR["metrics"],
        # "score": evaluatedR,
        "fields": fields,
        "img": img
    }
    return render(request,'overview.html',context=mydict)
    

def uploadView(request):
    return 
from django.forms.models import model_to_dict

def addCsvView(request):
    # save_dir = 'saved_models\\P\\ccf_220_keras_gru_static\\1'
    # print(os.path.join(settings.BASE_DIRV,save_dir,"wts"))
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

