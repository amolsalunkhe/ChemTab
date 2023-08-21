# set TF GPU memory growth so that it doesn't hog everything at once
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow import keras
from tensorflow.keras import layers as L
from optuna_train import *

# {'dataSetMethod': 'AllSpecies_randomequaltraintestsplit_AllDependants', 'ipscaler': None, 'opscaler': 'StandardScaler', 'noOfCpv': 12, 'ZmixPresent': 'Y', 'concatenateZmix': 'Y', 'kernel_constraint': 'Y', 'kernel_regularizer': 'Y', 'activity_regularizer': 'N', 'input_data_cols': ['YiCH4', 'YiH', 'YiO', 'YiO2', 'YiOH', 'YiH2O', 'YiHO2', 'YiH2O2', 'YiC', 'YiCH', 'YiCH2', 'YiCH2(S)', 'YiCH3', 'YiH2', 'YiCO', 'YiCO2', 'YiHCO', 'YiCH2O', 'YiCH2OH', 'YiCH3O', 'YiCH3OH', 'YiC2H', 'YiC2H2', 'YiC2H3', 'YiC2H4', 'YiC2H5', 'YiC2H6', 'YiHCCO', 'YiCH2CO', 'YiHCCOH', 'YiN', 'YiNH', 'YiNH2', 'YiNH3', 'YiNNH', 'YiNO', 'YiNO2', 'YiN2O', 'YiHNO', 'YiCN', 'YiHCN', 'YiH2CN', 'YiHCNN', 'YiHCNO', 'YiHOCN', 'YiHNCO', 'YiNCO', 'YiC3H7', 'YiC3H8', 'YiCH2CHO', 'YiCH3CHO', 'YiN2', 'YiAR'], 'val_losses': {'loss': -1.7983406782150269, 'dynamic_source_prediction_loss': -0.9774019122123718, 'static_source_prediction_loss': -0.8507312536239624, 'dynamic_source_prediction_R2_split': 0.9772778153419495, 'dynamic_source_prediction_source_pred_mean': 1.224331021308899, 'dynamic_source_prediction_source_true_mean': 1.7918648719787598, 'static_source_prediction_mae': 0.10440650582313538, 'static_source_prediction_mse': 0.15989357233047485, 'static_source_prediction_R2': 0.850985586643219, 'static_source_prediction_mape': 435.87158203125}, 'model_R2': 0.8515233500441262}

batch_size_exp=6

# you override these values based with the config values you pass (via dict.update())
cfg = {'zmix': 'Y', 'ipscaler': None, 'opscaler': 'StandardScaler', 'noOfCpv': 25,
       'activation': 'selu', 'width': 100, 'dropout_rate': 0.05, 'loss': 'mse', 
       'regressor_batch_norm': False, 'regressor_skip_connections': False, 'batch_norm_dynamic': False, #True,
       'kernel_constraint': 'N', 'kernel_regularizer': 'Y', 'activity_regularizer': 'N', 'batch_size': 2**batch_size_exp,
       'loss_weights': {'inv_prediction': 0.0, 'static_source_prediction': 1.0, 'dynamic_source_prediction': 1.0},
       'W_load_fn': '../datasets/Q_rot.csv.gz', 'use_dependants': False}
cfg['epochs'] = 1 # special because it should always be large
# NOTE: interestingly Amol's constraints are actually useful for dynamic prediction!
# This is because they stabilitize the Weight matrix which is too unstable during regular dynamic prediction

print(f'batch_size = 2^{batch_size_exp} = {cfg["batch_size"]}')

final_score = main(cfg)
print('model final (R^2) score:', final_score)
exprExec = main.exprExec
dm = exprExec.dm
exprExec.modelFactory.saveCurrModelAsBestModel('./long_train_best')
dm.save_PCA_data(fn='./long_train_best/PCA_data_long_train.csv')

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
