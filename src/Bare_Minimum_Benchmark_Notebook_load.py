#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn import gaussian_process
from sklearn.gaussian_process import GaussianProcessRegressor
import scipy.optimize
from sklearn.utils.optimize import _check_optimize_result
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
import time
from sklearn.decomposition import PCA, SparsePCA
import seaborn as sns
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_probability as tfp
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers, activations, initializers, constraints, Sequential
from tensorflow.keras.constraints import UnitNorm, Constraint
import sys


# ## Helpers

# In[2]:


from benchmark_model_code import * # Network Helpers
from benchmark_data_code import * # Data Helpers

import random
import numpy as np
#random.seed(0)
#np.random.seed(0)
#tf.random.set_seed(0)
halfData = getHalfData()

#error_df = pd.DataFrame(columns=['TotalAbsoluteError','TotalSquaredError','MeanAbsoluteError','MeanSquaredError','MeanPercentageError','NumPoints'])

control = int(sys.argv[1])

# ### Constrained DNN -- Baseline (Zmix + 4 Dim Linear Embedding; All Constraints)

# In[3]:

def build_model():
    encoding_dim = 4

    species_inputs = keras.Input(shape=(53,), name="species_input")

    Zmix = keras.Input(shape=(1,), name="zmix")

    x = layers.Dense(encoding_dim, activation="linear",kernel_constraint=UnitNorm(axis=0),kernel_regularizer=WeightsOrthogonalityConstraint(encoding_dim, weightage=1., axis=0),activity_regularizer=UncorrelatedFeaturesConstraint(encoding_dim, weightage=1.))(species_inputs)

    #Concatenate the Linear Embedding and Zmix together
    x = layers.Concatenate()([Zmix, x])

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    #Predict the source energy
    souener_pred = layers.Dense(1, name="prediction")(x)

    model = keras.Model(
        inputs=[species_inputs,Zmix],
        outputs=[souener_pred],
    )
    model.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(0.001))
    return model

#model = build_model()

# In[8]:


import pickle
with open('base_code_model/custom_objects.pickle', 'rb') as f:
    custom_objects = pickle.load(f)

def get_metrics():
    def log_mse(x,y): return tf.math.log(tf.math.reduce_mean((x-y)**2))
    def log_mae(x,y): return tf.math.log(tf.math.reduce_mean(tf.math.abs(x-y)))
    def exp_mse_mag(x,y): return tf.math.log(tf.math.reduce_mean((tf.math.exp(x)-tf.math.exp(y))**2))/tf.math.log(10.0)
    def exp_mae_mag(x,y): return tf.math.log(tf.math.reduce_mean(tf.math.abs(tf.math.exp(x)-tf.math.exp(y))))/tf.math.log(10.0)
    def R2(yt,yp): return 1-tf.math.reduce_mean((yp-yt)**2)/(tf.math.reduce_std(yt)**2)
    def exp_R2(yt,yp): # these are actual names above is for convenience
        return R2(tf.math.exp(yt), tf.math.exp(yp))
    return locals()
metrics = get_metrics()
custom_objects.update(metrics)

#model = keras.models.load_model('base_code_model/base_code_model.h5', custom_objects = custom_objects)
#model.summary(expand_nested=True)


# In[28]:


error_df = pd.DataFrame(columns=['TAE','TSE','MAE','MSE','MAPE','#Pts'])
   
normalized_species_train = halfData["normalized_species_train"]
Zmix_train = halfData["Zmix_train"] 
normalized_souener_train = halfData["normalized_souener_train"]
    
def train_model(model, pretrained=False):
    global error_df
    if not pretrained:
        history = model.fit([normalized_species_train,Zmix_train], 
                             normalized_souener_train,
                             validation_split=0.2,
                             verbose=1, 
                             epochs=100,
                             batch_size=32,
                             callbacks=[es])
        plot_loss(history)
    
    # data prep
    normalized_species_test = halfData["normalized_species_test"]
    Zmix_test =  halfData["Zmix_test"]
    
    normalized_souener_pred = model.predict([normalized_species_test,Zmix_test])
    
    # data prep
    scaler_souener = halfData["scaler_souener"]
    Y_pred = scaler_souener.inverse_transform(normalized_souener_pred)
    Y_pred = Y_pred.flatten()
    Y_test = halfData["Y_test"]
    
    err = {k: v for k,v in zip(error_df.columns, computeAndPrintError(Y_pred, Y_test))}
    return err

import os
os.system('mkdir NB_trained_models')
control=0
pretrained=1

for i in range(150):
    if not pretrained:
        if control: 
            print('control!')
            model = build_model()
        else:
            print('not control!') 
            model = keras.models.load_model(f'base_code_model/base_code_model{i}.h5', custom_objects = custom_objects)
    else:
        print('pretrained!')
        model = keras.models.load_model(f'trained_base_models/model{i}.h5', custom_objects = custom_objects)
    print(model.summary(expand_nested=True))
    err = train_model(model, pretrained=pretrained)
    #model.save(f'NB_trained_models/model{i}.h5')
    error_df = error_df.append(err, ignore_index=True)
    print(error_df.describe())

#error_df.to_csv(f'BM_train_err.csv')
#error_df.to_csv(f'BM_control={control}.csv')
error_df.to_csv(f'BM_pretrained.csv')
