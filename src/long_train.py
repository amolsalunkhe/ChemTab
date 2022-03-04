"""
Created on Wed Aug  4 17:50:06 2021

@author: amol
"""

# set TF GPU memory growth so that it doesn't hog everything at once
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow import keras
from tensorflow.keras import layers as L
from .main import *

# util for getting objects' fields' names
field_names = lambda x: list(vars(x).keys())

#Prepare the DataFrame that will be used downstream
dp = DataPreparer()
dp.createPCAs()
dp.sparsePCAs()
dp.zmixOrthogonalPCAs()
df = dp.getDataframe()

# currently passing dp eventually we want to abstract all the constants into 1 class
dm = DataManager(df, dp)
dm.train_portion = 0.8

""" prepare PCDNNV2 for loading (from prior experiments) """
exprExec = PCDNNV2ExperimentExecutor()
exprExec.setModelFactory(PCDNNV2ModelFactory())

dataType = 'randomequaltraintestsplit' #'frameworkincludedtrainexcludedtest'
inputType = 'AllSpecies'
dependants = 'NoDependants'
dataSetMethod = f'{inputType}_{dataType}_{dependants}'
ipscaler=opscaler = "MaxAbsScaler"#"MinMaxScaler" #'PositiveLogNormal'
ZmixPresent = 'N'
concatenateZmix = 'Y' if ZmixPresent=='Y' else 'N'
kernel_constraint = 'N'
kernel_regularizer = 'N'
activity_regularizer = 'N'
noOfCpv = 4
noOfNeurons = 53

exprExec.modelFactory.loss='mae'
exprExec.modelFactory.activation_func='relu'
exprExec.modelFactory.width=512
exprExec.modelFactory.dropout_rate=0#.5
exprExec.debug_mode = False
exprExec.batch_size = 8192
exprExec.epochs_override = 300000
exprExec.n_models_override = 1
exprExec.use_dependants = True
exprExec.use_dynamic_pred = True
#exprExec.min_mae = -float('inf')

# initialize experiment executor...
exprExec.dm = dm
exprExec.df_experimentTracker = pd.DataFrame()
exprExec.modelType = 'PCDNNV2'

# this will save the model as the best (since it starts with min_mae=-inf), but that is ok because it will also be the best
assert exprExec.epochs_override >= 10000 # ensure this model is the best!
history = exprExec.executeSingleExperiment(noOfNeurons,dataSetMethod,dataType,inputType,ZmixPresent,noOfCpv,concatenateZmix,kernel_constraint,
                                            kernel_regularizer,activity_regularizer,opscaler=opscaler, ipscaler=ipscaler)
dm.save_PCA_data(fn='PCA_data_long_train.csv')
#df.to_csv('PCA_data.csv', index=False)

#import matplotlib.pyplot as plt
#
##  "Accuracy"
#plt.plot(history.history['R2'])
#plt.plot(history.history['val_R2'])
#plt.title('model R2 history')
#plt.ylabel('R2')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.savefig('long_train_R2.png')


def build_mass_fraction_model(n_species=53):
    mass_fraction_pred = keras.models.Sequential()
    #layer_sizes = [32,64,128,256,512,256,128,64,32]
    layer_sizes = [16, 32, 64, 64] # TODO: use less layers
    for size in layer_sizes:
        mass_fraction_pred.add(L.Dense(size, activation='relu'))
    mass_fraction_pred.add(L.Dense(n_species))
    mass_fraction_pred.compile(optimizer='adam',loss='mse', metrics='mae')
    return mass_fraction_pred

data = dm.df
input_data = data[[f'PCDNNV2_PCA_{i+1}' for i in range(noOfCpv)]]
output_data = data[[c for c in data.columns if c.startswith('Yi')]]
mass_fraction_pred = build_mass_fraction_model(noOfNeurons)
mass_fraction_pred.fit(input_data, output_data, epochs=1000, validation_split=0.2)

mass_fraction_pred.save('mass_fraction_pred_model.h5')
