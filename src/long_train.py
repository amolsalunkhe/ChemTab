# set TF GPU memory growth so that it doesn't hog everything at once
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from optuna_train import *

# you override these values based with the config values you pass (via dict.update())
cfg = {'zmix': 'Y', 'ipscaler': None, 'opscaler': 'StandardScaler',
       'noOfCpv': 10, 'loss': 'R2', 'activation': 'selu', 'width': 800, 'dropout_rate': 0.0, 
       'regressor_batch_norm': False, 'regressor_skip_connections': False, 'batch_norm_dynamic': True,
       'kernel_constraint': 'Y', 'kernel_regularizer': 'Y', 'activity_regularizer': 'N', 'batch_size': 500,
       'loss_weights': {'static_source_prediction': 3.0, 'dynamic_source_prediction': 1.0}}
cfg['epochs'] = 5000 # special because it should always be large
cfg['use_L1_constrained_inversion']=True

final_score = main(cfg)
print('model final (R^2) score:', final_score)
exprExec = main.exprExec
dm = exprExec.dm
exprExec.modelFactory.saveCurrModelAsBestModel('./long_train_best')
dm.save_PCA_data(fn='./long_train_best/PCA_data_long_train.csv')

#from tensorflow import keras
#from tensorflow.keras import layers as L
#def build_mass_fraction_model(n_species=53):
#    mass_fraction_pred = keras.models.Sequential()
#    #layer_sizes = [32,64,128,256,512,256,128,64,32]
#    layer_sizes = [16, 32, 64, 64] # TODO: use less layers
#    for size in layer_sizes:
#        mass_fraction_pred.add(L.Dense(size, activation='relu'))
#    mass_fraction_pred.add(L.Dense(n_species))
#    mass_fraction_pred.compile(optimizer='adam',loss='mse', metrics='mae')
#    return mass_fraction_pred
#
#data = dm.df
#input_data = data[[f'PCDNNV2_PCA_{i+1}' for i in range(noOfCpv)]]
#output_data = data[[c for c in data.columns if c.startswith('Yi')]]
#mass_fraction_pred = build_mass_fraction_model(noOfNeurons)
#mass_fraction_pred.fit(input_data, output_data, epochs=1000, validation_split=0.2)
#
#mass_fraction_pred.save('mass_fraction_pred_model.h5')
