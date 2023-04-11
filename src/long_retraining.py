
"""
Created on Wed Aug 4 17:50:06 2021

@author: Amol & Dwyer
"""

# set TF GPU memory growth so that it doesn't hog everything at once
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import os
import numpy as np
from optuna_train import *

def print_scientific_notation(number):
    power = int(np.log(number)/np.log(10))
    print(f"Scientific Notation: {(loss/10**power)}*10^{power}")
    
import pandas as pd
assert pd.__path__[0]!='/opt/anaconda/lib/python3.8/site-packages/pandas', 'Error! You are using deprecated pandas package outside your conda environment. Did you use Jupyter Lab again?' 
# this is a nefarious problem with current version of anaconda, root cause is conda version install your own local one!
# lightly more superficial root cause is that you sometimes use jupyter lab which triggers you to use the /opt/anaconda/bin path backup when it sees jupyter lab isn't in local environment which breaks everything (uses outdated pandas)

raise NotImplemented('Hazard: This code is broken and gives fake results! (reloading the container module implicitly creates 2 weight matrices which can be trained independently rather than having them each reference the same. So naturally the results are fake.')

#Prepare the DataFrame that will be used downstream
dp = DataPreparer(fn=os.environ.setdefault('DATASET', ''))
df = dp.getDataframe()

# currently passing dp eventually we want to abstract all the constants into 1 class
dm = DataManager(df, dp)

""" prepare PCDNNV2 for loading (from prior experiments) """

exprExec = PCDNNV2ExperimentExecutor()
exprExec.debug_mode = False
exprExec.setModelFactory(PCDNNV2ModelFactory())

# for recording in custom objects dict
def get_metric_dict():
    def log_mse(x, y): return tf.math.log(tf.math.reduce_mean((x - y) ** 2))

    def log_mae(x, y): return tf.math.log(tf.math.reduce_mean(tf.math.abs(x - y)))

    def exp_mse_mag(x, y): return tf.math.log(
        tf.math.reduce_mean((tf.math.exp(x) - tf.math.exp(y)) ** 2)) / tf.math.log(10.0)

    def exp_mae_mag(x, y): return tf.math.log(
        tf.math.reduce_mean(tf.math.abs(tf.math.exp(x) - tf.math.exp(y)))) / tf.math.log(10.0)

    def R2(yt,yp): return tf.reduce_mean(1-tf.reduce_mean((yp-yt)**2, axis=0)/(tf.math.reduce_std(yt,axis=0)**2))

    def R2_inv(yt,yp): return R2(yt[:,1:], yp[:,1:])
    
    def exp_R2(yt, yp):  # these are actual names above is for convenience
        return R2(tf.math.exp(yt), tf.math.exp(yp))

    def source_true_mean(yt, yp):
        encoding_dim = yt.shape[1] // 2
        yt = yp[:, encoding_dim:]
        yp = yp[:, :encoding_dim]
        return tf.reduce_mean(yt, axis=-1)

    def source_pred_mean(yt, yp):
        encoding_dim = yt.shape[1] // 2
        yt = yp[:, encoding_dim:]
        yp = yp[:, :encoding_dim]
        return tf.reduce_mean(yp, axis=-1)

    def dynamic_source_loss(y_true, y_pred):
        assert y_true.shape[1] // 2 == y_true.shape[1] / 2
        encoding_dim = y_true.shape[1] // 2
        abs_diff = tf.math.abs(y_pred[:, :encoding_dim] - y_pred[:, encoding_dim:])
        return tf.reduce_mean(abs_diff)

    def R2_split(yt,yp):
        assert yp.shape[1]//2 == yp.shape[1]/2
        encoding_dim = yp.shape[1]//2
        yt=yp[:,:encoding_dim]
        yp=yp[:,encoding_dim:]
        # NOTES: verified that len(yt.shape)==2 and yt.shape[0] is batch_dim (not necessarily None)
        assert len(yp.shape)==2
        return tf.reduce_mean(1-tf.math.reduce_mean((yp-yt)**2, axis=0)/(tf.math.reduce_std(yt,axis=0)**2))
    return locals()

# fill globals with metric functions
globals().update(get_metric_dict())

bestModel, experimentSettings = exprExec.modelFactory.openBestModel()
opt = exprExec.modelFactory.getOptimizer()
losses={'static_source_prediction': lambda yt, yp: -R2(yt, yp), 'dynamic_source_prediction': lambda yt, yp: -R2_split(yt, yp)}
metrics = {'static_source_prediction': ['mae', 'mse', R2, 'mape', R2_inv],
           'dynamic_source_prediction': [R2_split, source_pred_mean, source_true_mean]}
# for metric definitions see get_metric_dict()

bestModel.compile(loss=losses, optimizer=opt, metrics=metrics)

exprExec.epochs_override = 100000
exprExec.batch_size=3000

# settings for training on full dataset (gets slightly better performance of course)
dm.train_portion=0.9
dm.createTrainTestData(experimentSettings['dataSetMethod'], experimentSettings['noOfCpv'], experimentSettings['ipscaler'], experimentSettings['opscaler'])
exprExec.dm = dm#experimentSettings['data_manager']

exprExec.model = bestModel
exprExec.use_dynamic_pred = True

exprExec.fitModelAndCalcErr(rebuild=False)
