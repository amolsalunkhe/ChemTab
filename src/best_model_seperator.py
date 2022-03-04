#!/bin/python3

# NOTE: this file is meant to replace model_seperator.py, 
# but for now original is kept as a backup 

import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys, os
import pandas as pd
from main import *

from models.pcdnnv2_model_factory import get_metric_dict 

dp = DataPreparer() #Prepare the DataFrame that will be used downstream
df = dp.getDataframe()
dm = DataManager(df, dp) # currently passing dp eventually we want to abstract all the constants into 1 class

exprExec = PCDNNV2ExperimentExecutor()
exprExec.setModelFactory(PCDNNV2ModelFactory())
bestModel, experimentSettings = exprExec.modelFactory.openBestModel()
assert experimentSettings['ipscaler']=='MaxAbsScaler' # we cannot center data, as it makes transform non-linear

dm.createTrainTestData(experimentSettings['dataSetMethod'], experimentSettings['noOfCpv'],
                       experimentSettings['ipscaler'], experimentSettings['opscaler'])

composite_model = bestModel#keras.models.load_model(sys.argv[1])
composite_model.summary()

decomp_dir = f'./PCDNNV2_decomp'
os.system(f'rm -r {decomp_dir}; mkdir {decomp_dir}')

get_layers_by_name = lambda model: {layer.name: layer for layer in model.layers}
layers_by_name = get_layers_by_name(composite_model)

linear_embedder = layers_by_name['linear_embedding']

# this integrates a batch norm layer & input_scaler into simple W mat & bias
def compute_simplified_W_and_b(linear_embedder, input_scaler):
    W_layer = linear_embedder.get_layer('linear_embedding')
    batch_layer = linear_embedder.get_layer('batch_norm')

    from sklearn.preprocessing import MaxAbsScaler

    weights = batch_layer.weights
    assert len(weights)==2
    assert type(input_scaler) is MaxAbsScaler

    W = W_layer.weights[0].numpy().T # transpose so this follows standard matrix conventions
    vec_bias = weights[0].numpy().reshape(-1,1)

    # compute W1
    sigma_scale_mat = np.diag(1/np.sqrt(weights[1])) # = sigma^-1
    W1 = W.dot(sigma_scale_mat)

    # use W1
    M_scale_mat = np.diag(1/np.abs(input_scaler.scale_)) # = |M|^-1
    W2 = W1.dot(M_scale_mat) # apply input scaling implicitly w/ matrix
    vec_bias = -W1.dot(vec_bias)

    W2 = W2.T # put W2 back in original weird keras format (e.g x.T*W+b=y)
    return W2, vec_bias.squeeze()

w, vec_bias = compute_simplified_W_and_b(linear_embedder, dm.inputScaler)

print(f'linear embedder weights shape: {w.shape}') # shape is [53, nCPV]

CPV_names = [f'CPV_{i}' for i in range(w.shape[1])]

if 'zmix' in layers_by_name:
    raise NotImplementedError('you need to get the actual weights for zmix here')
    zmix_weights = np.random.normal(size=(w.shape[0],1))
    w = np.concatenate([zmix_weights, w],axis=1) # checked on 10/10/21: that zmix comes first
    CPV_names = ['zmix'] + CPV_names

# include bias in weight matrix
w_full = np.concatenate((vec_bias.reshape(1,-1),w), axis=0)
weight_df = pd.DataFrame(w_full, index=['bias']+list(dm.input_data_cols), columns=CPV_names) # dm.input_data_cols is why we recreate training data
weight_df.to_csv(f'{decomp_dir}/weights.csv', index=True, header=True)

#np.savetxt(f'{decomp_dir}/weights.csv', w, delimiter=',')
#linear_embedder.save(f'{decomp_dir}/linear_embedding')

regressor = layers_by_name['regressor']
input_ = keras.layers.Input(shape=regressor.input_shape[1:], name='input_1')
wrapper = keras.models.Model(inputs=input_, outputs=regressor(input_))
#wrapper.add(keras.layers.Input(shape=regressor.input_shape[1:], name='input_1'))
#wrapper.add(regressor)
wrapper.save(f'{decomp_dir}/regressor')

globals().update(get_metric_dict())

losses={'static_source_prediction': 'mae', 'dynamic_source_prediction': dynamic_source_loss}
metrics={'static_source_prediction': ['mae', 'mse', R2], 'dynamic_source_prediction': [R2_split, source_pred_mean, source_true_mean]}
# for metric definitions see get_metric_dict()

X_CPVs_test_true = linear_embedder.predict(dm.X_scaled_test)

regressor.compile(loss=losses, optimizer='adam', metrics=metrics)
X_CPV_test = dm.X_test.dot(w)
X_CPV_test = X_CPV_test + vec_bias

sanity_R2 = 1-np.mean((X_CPVs_test_true-X_CPV_test)**2)/np.var(X_CPVs_test_true)
assert sanity_R2 > 0.99999
