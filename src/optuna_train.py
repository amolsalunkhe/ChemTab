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

from tensorflow import keras
from tensorflow.keras import layers as L
from copy import deepcopy
import optuna
import sys
import numpy as np
import pandas as pd

debug_mode = False #True
if debug_mode: print('debugging!', file=sys.stderr)

def main(cfg={}):
    # assert new cfg values are valid (must exist in default)
    assert all([k in main.default_cfg for k in cfg])
    
    # update default template with passed config
    full_cfg = deepcopy(main.default_cfg)
    full_cfg.update(cfg)
    cfg = full_cfg

    #Prepare the DataFrame that will be used downstream
    dp = DataPreparer(cfg['data_fn'])
    df = dp.getDataframe()
    
    # currently passing dp eventually we want to abstract all the constants into 1 class
    dm = DataManager(df, dp)
    dm.train_portion = cfg['train_portion']
    
    """ prepare PCDNNV2 for loading (from prior experiments) """
    exprExec = PCDNNV2ExperimentExecutor()
    exprExec.setModelFactory(PCDNNV2ModelFactory())
    
    dataType = 'randomequaltraintestsplit'
    inputType = 'AllSpecies'
    dependants = 'AllDependants' if cfg['use_dependants'] else 'SouenerOnly'
    dataSetMethod = f'{inputType}_{dataType}_{dependants}'
    ipscaler=cfg['ipscaler']
    opscaler=cfg['opscaler']
    assert opscaler!='QuantileTransformer' and ipscaler!='QuantileTransformer'
    ZmixPresent = cfg['zmix'] 
    concatenateZmix = 'Y' if ZmixPresent=='Y' else 'N'
    kernel_constraint = cfg['kernel_constraint']
    kernel_regularizer = cfg['kernel_regularizer']
    activity_regularizer = cfg['activity_regularizer']
    noOfCpv = cfg['noOfCpv']
    df = pd.read_csv(cfg['data_fn'])
    Yi_cols = [col for col in df.columns if col.startswith('Yi')]
    noOfNeurons = len(Yi_cols)
   
    exprExec.modelFactory.W_load_fn=cfg['W_load_fn']
    exprExec.modelFactory.loss=cfg['loss']
    exprExec.modelFactory.activation_func=cfg['activation']
    exprExec.modelFactory.width=cfg['width']
    exprExec.modelFactory.dropout_rate=cfg['dropout_rate']
    exprExec.modelFactory.regressor_skip_connections = cfg['regressor_skip_connections']
    exprExec.modelFactory.regressor_batch_norm = cfg['regressor_batch_norm']
    exprExec.modelFactory.batch_norm_dynamic_pred = cfg['batch_norm_dynamic']
    exprExec.modelFactory.loss_weights = cfg['loss_weights'] 
    exprExec.modelFactory.W_batch_norm = cfg['W_batch_norm']
    exprExec.modelFactory.starter_learning_rate = cfg['starter_learning_rate']
    exprExec.modelFactory.decay_steps = cfg['decay_steps']
    exprExec.modelFactory.decay_rate = cfg['decay_rate']
    exprExec.modelFactory.clip_grad_norm = cfg['clip_grad_norm']

    exprExec.debug_mode = False
    exprExec.batch_size = cfg['batch_size'] 
    exprExec.epochs_override = cfg['epochs'] 
    exprExec.n_models_override = cfg['n_models_override']
    exprExec.use_dynamic_pred = cfg['use_dynamic_pred']
    exprExec.use_dependants = dependants == 'AllDependants'
    assert (dependants == 'AllDependants') == cfg['use_dependants']
    #exprExec.min_mae = -float('inf')
    
    # initialize experiment executor...
    exprExec.dm = dm
    exprExec.df_experimentTracker = pd.DataFrame()
    exprExec.modelType = 'PCDNNV2'
    
    final_score = exprExec.executeSingleExperiment(noOfNeurons,dataSetMethod,dataType,inputType,ZmixPresent,noOfCpv,concatenateZmix,kernel_constraint,
                                                   kernel_regularizer,activity_regularizer,opscaler=opscaler, ipscaler=ipscaler)
    
    print(final_score)
    main.exprExec = exprExec # this may be of interest to the caller
    return final_score - noOfCpv*0.001 # small penalty for using more CPVs...

# you override these values based with the config values you pass (via dict.update())
main.default_cfg = {'ipscaler': None, 'opscaler': 'StandardScaler', 'noOfCpv': 10, 'loss': 'R2',
                    'activation': 'selu', 'width': 2048, 'dropout_rate': 0.0,
                    'batch_size': 500, 'activity_regularizer': 'N', 'kernel_regularizer': 'N', 'kernel_constraint': 'N', 
                    'loss_weights': {'inv_prediction': 1.0, 'static_source_prediction': 1.0, 'dynamic_source_prediction': 1.0},
                    'regressor_batch_norm': False, 'regressor_skip_connections': False, 'W_load_fn': None}

# add optimizer hyper-parameters:
default_opt_hparams = {'starter_learning_rate': 0.0001, 'decay_steps': 100000, 'decay_rate': 0.96, 'clip_grad_norm': 2.5}
main.default_cfg.update(default_opt_hparams)

constants = {'epochs': 10 if debug_mode else 500, 'train_portion': 0.7, 'n_models_override': 1, 'zmix': 'Y',
             'use_dynamic_pred': True, 'use_dependants': True, 'use_tensorboard': False, 'data_fn': os.environ.setdefault('DATASET', ''),
             #'kernel_constraint': 'Y', 'kernel_regularizer': 'Y',  
             'W_batch_norm': False, 'batch_norm_dynamic': False} #, 'ipscaler': None} # this line is all garbage configs
main.default_cfg.update(constants)
# add variables generally held as constant

def auto_spread(trial, default_cfg, spread=0.5):
    import math
    trial_cfg = {}
    for name,val  in default_cfg.items():
        if type(val) is float:
            trial_cfg[name]=trial.suggest_float(name, *sorted([math.exp(math.log(val)*spread), math.exp(math.log(val)*(1+spread))]))
        elif type(val) is int:
            trial_cfg[name]=trial.suggest_int(name, int(val*spread), int(val*(1+spread)))
        elif type(val) is bool:
            trial_cfg[name]=trial.suggest_categorical(name, [True, False])
        else: trial_cfg[name]=default_cfg[name] # for e.g. categorical strings! 
    return trial_cfg

# wrapper for main use in optuna (protects against crashes & uses trials to populate cfg dict)
import traceback
def main_safe(trial=None):
    cfg = {} # (no changes default config)

    if trial:
        scalers_types = [None, 'MinMaxScaler', 'MaxAbsScaler', 'StandardScaler', 'RobustScaler']#,'QuantileTransformer']

        cfg = {'ipscaler': trial.suggest_categorical('input_scaler', scalers_types),
               'opscaler': trial.suggest_categorical('output_scaler', scalers_types), 'noOfCpv': trial.suggest_int('noOfCpv', *[6, 20]),
               'loss': trial.suggest_categorical('loss', ['mae', 'mse', 'R2', 'mape']), 'activation': trial.suggest_categorical('activation', ['tanh', 'selu', 'relu']),
               'width': trial.suggest_int('width', *[1024, 4096]), 'dropout_rate': trial.suggest_float('dropout_rate', *[0, 0.4]),
               'regressor_batch_norm': trial.suggest_categorical('regressor_batch_norm', [True, False]),
               'regressor_skip_connections': trial.suggest_categorical('regressor_skip_connections', [True, False]),
               'activity_regularizer': trial.suggest_categorical('activity_regularizer', ['Y', 'N']), 'batch_size': trial.suggest_int('batch_size', *[128, 1028]),
               'kernel_regularizer': trial.suggest_categorical('kernel_regularizer', ['Y', 'N']), 'kernel_constraint': trial.suggest_categorical('kernel_constraint', ['Y', 'N']),
               'loss_weights': {'static_source_prediction': trial.suggest_float('static_loss_weight', *[0.1, 10.0]),
                                'inv_prediction': trial.suggest_float('inv_loss_weight', *[0.1, 10.0]),
                                'dynamic_source_prediction': trial.suggest_float('dynamic_loss_weight', *[0.1, 10.0])}} 

 
        # add optimizer hyper-parameters:
        #manual_opt_bounds = {'starter_learning_rate': trial.suggest_float('starter_learning_rate': [0, 0.1]), 'decay_steps': trial.suggest_int('decay_steps': [10000, 100000]),
        #'decay_rate': trial.suggest_float('decay_rate': [0, 0.1]), 'clip_grad_norm': trial.suggest_float('clip_grad_norm': [1.0,5.0)}
        #print('manual_opt_bounds: ', manual_opt_bounds)
        auto_opt_sample = auto_spread(trial, default_opt_hparams, spread=0.5)
        cfg.update(auto_opt_sample)
        print('auto_opt_sample: ', auto_opt_sample) 

        global constants
        constant_names, cfg_names = set(constants.keys()), set(cfg.keys())
        overlapping_names = constant_names.intersection(cfg_names)
        if len(overlapping_names): raise RuntimeError(f'Invalid config, these constants: {",".join(overlapping_names)} are being overridden!') 
    try:
        return main(cfg)
    except:
        print('Main caught exception:', file=sys.stderr)
        traceback.print_exc()
        print('offending config:', cfg, flush=True)
        raise optuna.exceptions.TrialPruned() # prune this trial if there is an exception!

if __name__ == '__main__':
    import os
    import random
    os.system(f'mv models/best_models/PCDNNV2Model models/best_models/PCDNNV2Model-{random.random()}')
    # we should start with a fresh slate because usually settings change across optuna runs

    if not os.path.exists(os.environ['DATASET']):
        err_msg = f"DATASET: {os.environ['DATASET']} does not exist!\n Here are valid files in the chosen directory: {', '.join(os.listdir(os.path.dirname(os.environ['DATASET'])))}"
        raise RuntimeError(err_msg)

    study_name = os.environ.setdefault('STUDY_NAME', 'optuna')
    study = optuna.create_study(study_name=study_name, direction='maximize')
    study.optimize(main_safe, n_trials=5 if debug_mode else 1000)
    import pickle
    with open(f'{study_name}_study.pickle', 'wb') as f:
        pickle.dump(study, f)

    print('='*50)
    print('study:', study_name, 'complete!')
    print('best value:', study.best_value)
    print('best params:', study.best_params)  # E.g. {'x': 2.002108042}
    print('='*50)

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
