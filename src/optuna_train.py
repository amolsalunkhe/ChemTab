# set TF GPU memory growth so that it doesn't hog everything at once
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow import keras
from tensorflow.keras import layers as L
from copy import deepcopy
from main import *
import optuna
import sys

# util for getting objects' fields' names
field_names = lambda x: list(vars(x).keys())

debug_mode = True 
if debug_mode: print('debugging!', file=sys.stderr)

def main(cfg={}):
    # assert new cfg values are valid (must exist in default)
    assert all([k in main.default_cfg for k in cfg])
    
    # update default template with passed config
    full_cfg = deepcopy(main.default_cfg)
    full_cfg.update(cfg)
    cfg = full_cfg

    #Prepare the DataFrame that will be used downstream
    dp = DataPreparer()
    dp.createPCAs()
    dp.sparsePCAs()
    dp.zmixOrthogonalPCAs()
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
    noOfNeurons = 53
    
    exprExec.modelFactory.loss=cfg['loss']
    exprExec.modelFactory.activation_func=cfg['activation']
    exprExec.modelFactory.width=cfg['width']
    exprExec.modelFactory.dropout_rate=cfg['dropout_rate']
    exprExec.modelFactory.use_R2_losses = exprExec.modelFactory.loss=='R2'
    exprExec.modelFactory.batch_norm_dynamic_pred = cfg['batch_norm_dynamic']
    exprExec.modelFactory.loss_weights = cfg['loss_weights'] 
    exprExec.modelFactory.W_batch_norm = cfg['W_batch_norm']
        
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
    
    # this will save the model as the best (since it starts with min_mae=-inf), but that is ok because it will also be the best
    #assert exprExec.epochs_override >= 10000 # ensure this model is the best!
    history = exprExec.executeSingleExperiment(noOfNeurons,dataSetMethod,dataType,inputType,ZmixPresent,noOfCpv,concatenateZmix,kernel_constraint,
                                               kernel_regularizer,activity_regularizer,opscaler=opscaler, ipscaler=ipscaler)
    history = history.history
    
    # to compute 'final R^2 score', we take the exponentially weighted average of the 2 val_R2 metrics then choose the lowest one (pessimistic estimate)
    val_R2s = [0,0]
    beta = 0.7
    for val_R2_split, val_R2 in zip(history['val_dynamic_source_prediction_R2_split'], history['val_emb_and_regression_model_R2']):
        val_R2s[0]=val_R2s[0]*(1-beta) + val_R2_split*beta
        val_R2s[1]=val_R2s[1]*(1-beta) + val_R2*beta
    final_score = min(val_R2s)
    print(final_score)
    print(val_R2s)
    print(history)
    main.exprExec = exprExec # this may be of interest to the caller
    return final_score

# you override these values based with the config values you pass (via dict.update())
main.default_cfg = {'zmix': 'N', 'ipscaler': None, 'opscaler': 'MinMaxScaler', 'noOfCpv': 4, 'loss': 'R2',
                    'activation': 'selu', 'width': 512, 'dropout_rate': 0.0, 'batch_norm_dynamic': False,
                    'kernel_constraint': 'N', 'kernel_regularizer': 'N', 'activity_regularizer': 'N', 'batch_size': 256,
                    'loss_weights': {'static_source_prediction': 1.0, 'dynamic_source_prediction': 1.0}}
main.default_cfg.update({'epochs': 1 if debug_mode else 500, 'train_portion': 0.8, 'n_models_override': 1,
                         'use_dynamic_pred': True, 'use_dependants': True, 'W_batch_norm': False})
# add variables generally held as constant

# wrapper for main use in optuna (protects against crashes & uses trials to populate cfg dict)
import traceback
def main_safe(trial=None):
    # default config
    cfg = {}

    if trial:
        scalers_types = [None, 'MinMaxScaler', 'MaxAbsScaler', 'StandardScaler', 'RobustScaler']#,'QuantileTransformer']

        cfg = {'zmix': trial.suggest_categorical('zmix', ['Y', 'N']), 'ipscaler': trial.suggest_categorical('input_scaler', scalers_types),
               'opscaler': trial.suggest_categorical('output_scaler', scalers_types), 'noOfCpv': trial.suggest_int('noOfCpv', *[3, 10]),
               'loss': trial.suggest_categorical('loss', ['mae', 'mse', 'R2']), 'activation': trial.suggest_categorical('activation', ['selu', 'relu']),
               'width': trial.suggest_int('width', *[256, 1024]), 'dropout_rate': trial.suggest_float('dropout_rate', *[0, 0.4]),
               'batch_norm_dynamic': trial.suggest_categorical('batch_norm_dynamic', [True, False]),
               'kernel_constraint': trial.suggest_categorical('kernel_constraint', ['Y', 'N']), 'kernel_regularizer': trial.suggest_categorical('kernel_regularizer', ['Y', 'N']),
               'activity_regularizer': trial.suggest_categorical('activity_regularizer', ['Y', 'N']), 'batch_size': trial.suggest_int('batch_size', *[128, 1028]),
               'loss_weights': {'static_source_prediction': trial.suggest_float('static_loss_weight', *[0.1, 10.0]),
                                'dynamic_source_prediction': trial.suggest_float('dynamic_loss_weight', *[0.1, 10.0])}} 
    try:
        return main(cfg)
    except:
        print('Main caught exception:', file=sys.stderr)
        traceback.print_exc()
        print('offending config:', cfg, flush=True)
        raise optuna.exceptions.TrialPruned() # prune this trial if there is an exception!

if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(main_safe, n_trials=5 if debug_mode else 500)
    import pickle
    with open('study.pickle', 'wb') as f:
        pickle.dump(study, f)
    
    print('best params:')
    print(study.best_params)  # E.g. {'x': 2.002108042}

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