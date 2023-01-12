#!/bin/python3

import os  # this enables XLA optimized computations
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import tensorflow as tf
from tensorflow import keras

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from data.pre_processing import DataPreparer
from data.train_test_manager import DataManager
from experiment_executor.pcdnn_v2_experiment_executor import PCDNNV2ExperimentExecutor
from models.pcdnnv2_model_factory import PCDNNV2ModelFactory

import numpy as np
import sys, os
import pandas as pd
import sklearn.linear_model
from fit_rescaling_layer import fit_rescaling_layer
import yaml

decomp_dir = f'./PCDNNV2_decomp'
os.system(f'rm -r {decomp_dir}; mkdir {decomp_dir}')

exprExec = PCDNNV2ExperimentExecutor()
exprExec.setModelFactory(PCDNNV2ModelFactory())
bestModel, experimentSettings = exprExec.modelFactory.openBestModel()
assert experimentSettings['ipscaler']==None # we cannot center data, as it makes transform non-linear
bestModel = exprExec.modelFactory.getEmbRegressor() # shed container model
dm = experimentSettings['data_manager']

del experimentSettings['data_manager']
del experimentSettings['history']
with open(f'{decomp_dir}/experiment_record.yaml', 'w') as f:
	yaml.dump(experimentSettings, f)

composite_model = bestModel
composite_model.summary()

get_layers_by_name = lambda model: {layer.name: layer for layer in model.layers}
layers_by_name = get_layers_by_name(composite_model)

linear_embedder = layers_by_name['linear_embedding']
w = np.asarray(get_layers_by_name(linear_embedder)['linear_embedding'].weights[0])
print(f'linear embedder weights shape: {w.shape}') # shape is [53, nCPV]

def add_unit_L1_layer_constraint(x, first_n_preserved=1, layer_name=None):
    """ this is designed to constraint the [first_n_preserved:]
    outputs from the inversion layer to fall between 0 & 1 and to sum to 1 """
    from tensorflow.keras import layers
    
    inputs_ = layers.Input((x.shape[1]-first_n_preserved,))
    out = tf.math.maximum(inputs_, 0)
    out = tf.math.minimum(out, 1)
    
    out = out/tf.math.reduce_sum(out, axis=-1, keepdims=True)
    model = keras.models.Model(inputs=inputs_, outputs=out, name='Unit_L1_constraint')
    return layers.Concatenate(axis=-1, name=layer_name)([x[:,:first_n_preserved], model(x[:,first_n_preserved:])])

# verified to work 6/20/22
def derive_Zmix_weights(df):
    import sklearn.linear_model
    Yi_cols = [col for col in df.columns if col.startswith('Yi')]
    X_data = df[Yi_cols]
    Y_data = df['Zmix']

    # verified (empirically) that it is ok to not use intercept 12/27/22
    lm = sklearn.linear_model.LinearRegression(fit_intercept=False, positive=True)
    lm.fit(X_data, Y_data)
    assert lm.score(X_data, Y_data)>0.95 # assert R^2=1 (since zmix should be linear combinatino of Yi's)
    Zmix_weights = pd.Series({k: v for k,v in zip(X_data.columns, lm.coef_)}) 
    return Zmix_weights # seems more reasonable to return a series, even though it is later needed as DF...
    #return pd.DataFrame(Zmix_weights, columns=['Zmix'])

# verified to work 5/24/22
def get_weight_inv_df(weights_df):
    weights_array = np.asarray(weights_df)
    weights_inv = np.linalg.pinv(weights_array)
    weights_inv_df = pd.DataFrame(data=weights_inv, columns=weights_df.index, index=weights_df.columns)
    return weights_inv_df

# convert to df to prepare for csv saving
CPV_names = [f'CPV_{i}' for i in range(w.shape[1])]
weight_df = pd.DataFrame(w, index=dm.input_data_cols, columns=CPV_names) # dm.input_data_cols is why we recreate training data

# zmix is now required for ablate integration!
assert 'zmix' in layers_by_name

# save pseudo inverse for Varun
#weight_inv_df = get_weight_inv_df(weight_df)
#weight_inv_df.to_csv(f'{decomp_dir}/weights_inv.csv', index=True, header=True)
Zmix_weights = derive_Zmix_weights(dm.df)
Zmix_weights = pd.DataFrame(Zmix_weights, columns=['zmix'])
weight_df = pd.concat([Zmix_weights, weight_df], axis=1)

weight_df.to_csv(f'{decomp_dir}/weights.csv', index=True, header=True)
linear_embedder.save(f'{decomp_dir}/linear_embedding') # usually not needed but included for completeness

# give regressor special input name that works with cpp tensorflow
regressor = layers_by_name['regressor']
input_ = keras.layers.Input(shape=regressor.input_shape[1:], name='input_1')
output = regressor(input_)

rescaling_layer, (m,b) = fit_rescaling_layer(dm.outputScaler, layer_name='static_source_prediction')

## m & b from y=mx+b
output['static_source_prediction'] = rescaling_layer(output['static_source_prediction'])
#output['static_source_prediction'] = add_unit_L1_layer_constraint(output['static_source_prediction'], 1, layer_name='static_source_prediction')
output['dynamic_source_prediction'] = keras.layers.Rescaling(1, name='dynamic_source_prediction')(output['dynamic_source_prediction']) # identity layer to rename properly
wrapper = keras.models.Model(inputs=input_, outputs=output, name='regressor')

wrapper.save(f'{decomp_dir}/regressor')
import adapt_test_targets # this will automatically build/save test targets for use by ablate

#scaling_params = np.stack([m,b])
#np.savetxt(f'{decomp_dir}/scaling_params.txt', scaling_params)
