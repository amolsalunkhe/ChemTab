#!/bin/python3

# NOTE: this file is meant to replace model_seperator.py, 
# but for now original is kept as a backup 

import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys, os
import pandas as pd
from main import *


dp = DataPreparer() #Prepare the DataFrame that will be used downstream
df = dp.getDataframe()
dm = DataManager(df, dp) # currently passing dp eventually we want to abstract all the constants into 1 class

exprExec = PCDNNV2ExperimentExecutor()
exprExec.setModelFactory(PCDNNV2ModelFactory())
bestModel, experimentSettings = exprExec.modelFactory.openBestModel()
dm.createTrainTestData(experimentSettings['dataSetMethod'], experimentSettings['noOfCpv'],
                       experimentSettings['ipscaler'], experimentSettings['opscaler'])

composite_model = bestModel#keras.models.load_model(sys.argv[1])
composite_model.summary()

decomp_dir = f'./PCDNNV2_decomp'
os.system(f'rm -r {decomp_dir}; mkdir {decomp_dir}')

get_layers_by_name = lambda model: {layer.name: layer for layer in model.layers}
layers_by_name = get_layers_by_name(composite_model)

linear_embedder = layers_by_name['linear_embedding']
w = np.asarray(get_layers_by_name(linear_embedder)['linear_embedding'].weights[0])
print(f'linear embedder weights shape: {w.shape}') # shape is [53, nCPV]

CPV_names = [f'CPV_{i}' for i in range(w.shape[1])]

if 'zmix' in layers_by_name:
    raise NotImplementedError('you need to get the actual weights for zmix here')
    zmix_weights = np.random.normal(size=(w.shape[0],1))
    w = np.concatenate([zmix_weights, w],axis=1) # checked on 10/10/21: that zmix comes first
    CPV_names = ['zmix'] + CPV_names
weight_df = pd.DataFrame(w, index=dm.input_data_cols, columns=CPV_names) # dm.input_data_cols is why we recreate training data
weight_df.to_csv(f'{decomp_dir}/weights.csv', index=True, header=True)

#np.savetxt(f'{decomp_dir}/weights.csv', w, delimiter=',')
#linear_embedder.save(f'{decomp_dir}/linear_embedding')

regressor = layers_by_name['regressor']
input_ = keras.layers.Input(shape=regressor.input_shape[1:], name='input_1')
wrapper = keras.models.Model(inputs=input_, outputs=regressor(input_))
#wrapper.add(keras.layers.Input(shape=regressor.input_shape[1:], name='input_1'))
#wrapper.add(regressor)
wrapper.save(f'{decomp_dir}/regressor')
