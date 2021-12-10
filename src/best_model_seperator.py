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

#print(sys.argv[1])
composite_model = bestModel#keras.models.load_model(sys.argv[1])
composite_model.summary()

decomp_dir = f'./PCDNNV2_decomp'
os.system(f'mkdir {decomp_dir}')

get_layer = lambda model, name: [layer for layer in model.layers if layer.name==name][0]

linear_embedder = get_layer(composite_model, 'linear_embedding')
w = np.asarray(get_layer(linear_embedder, 'linear_embedding').weights[0])
print(f'linear embedder weights shape: {w.shape}') # shape is [53, nCPV]

weight_df = pd.DataFrame(w)
weight_df['VarName'] = dm.input_data_cols
weight_df.to_csv(f'{decomp_dir}/weights.csv', index=False)

#np.savetxt(f'{decomp_dir}/weights.csv', w, delimiter=',')
#linear_embedder.save(f'{decomp_dir}/linear_embedding')

regressor = get_layer(composite_model, 'prediction')
regressor.save(f'{decomp_dir}/regressor')
