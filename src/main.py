# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 17:50:06 2021

@author: amol
"""

import os # this enables XLA optimized computations
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from data.pre_processing import DataPreparer 
from data.train_test_manager import DataManager 
from experiment_executor.gp_experiment_executor import GPExperimentExecutor
from experiment_executor.simple_dnn_experiment_executor import DNNExperimentExecutor
from experiment_executor.pcdnn_v1_experiment_executor import PCDNNV1ExperimentExecutor
from experiment_executor.pcdnn_v2_experiment_executor import PCDNNV2ExperimentExecutor

from models.gp_model_factory import GPModel
from models.simplednn_model_factory import SimpleDNNModelFactory
from models.pcdnnv1_model_factory import PCDNNV1ModelFactory
from models.pcdnnv2_model_factory import PCDNNV2ModelFactory
import pandas as pd


def compile_results(out_folder='data_results'):
    print('Organizing results...')
    os.system(f'rm -r {out_folder} {out_folder}.zip')
    os.system(f'mkdir {out_folder}')
    os.system(f'mv PCA_data.csv ./{out_folder}/.')
    os.system(f'mv *_Experiment_Results.csv ./{out_folder}/.')
    os.system(f'cp -r ./models/best_models ./{out_folder}/.')
    os.system(f'zip -r {out_folder}.zip {out_folder}; rm -r {out_folder}')
    print('Done!')

def run_gp_experiments(dm, debug_mode = False):
    '''
    TODO: search for '#TODO:uncomment' in the 'experiment_executor/gp_experiment_executor.py' uncomment & comment out the necessary lines
    '''
    gp = GPModel()
    #experimentTrackingFields = ['Model','Dataset','Cpv Type','#Cpv',"ZmixExists",'MAE','TAE','MSE','TSE','#Pts','FitTime','PredTime']
    df_experimentTracker = pd.DataFrame()#columns=experimentTrackingFields)
    
    expExectr = GPExperimentExecutor()
    expExectr.debug_mode = debug_mode
    expExectr.setModel(gp.getModel("Matern"))
    expExectr.executeExperiments(dm, "GP_Matern", df_experimentTracker)
    
    expExectr = GPExperimentExecutor()
    expExectr.debug_mode = debug_mode
    expExectr.setModel(gp.getModel("RationalQuadratic"))
    expExectr.executeExperiments(dm, "GP_RationalQuadratic", df_experimentTracker)
    
    expExectr = GPExperimentExecutor()
    expExectr.debug_mode = debug_mode
    expExectr.setModel(gp.getModel("Matern_RationalQuadratic"))
    expExectr.executeExperiments(dm, "GP_Matern_RationalQuadratic", df_experimentTracker)

    df_experimentTracker.to_csv('GP_Experiment_Results.csv', sep='\t',encoding='utf-8', index=False)
    print(df_experimentTracker.describe())
    
    return expExectr

def run_simple_dnn_experiments(dm, debug_mode = False):
    '''
    TODO: search for '#TODO:uncomment' in the 'experiment_executor/simple_dnn_experiment_executor.py' uncomment & comment out the necessary lines
    '''
    #dnnexperimentTrackingFields = ['Model','Dataset','Cpv Type','#Cpv',"ZmixExists",'MAE','TAE','MSE','TSE','#Pts','FitTime','PredTime','MAX-MAE','MAX-TAE','MAX-MSE','MAX-TSE','MIN-MAE','MIN-TAE','MIN-MSE','MIN-TSE']


    expExectr = DNNExperimentExecutor()
    expExectr.debug_mode = debug_mode

    expExectr.setModelFactory(SimpleDNNModelFactory())
    expExectr.executeExperiments(dm, "Simple_DNN", pd.DataFrame())

    df_experimentTracker = expExectr.df_experimentTracker
    df_experimentTracker.to_csv('SimpleDNN_Experiment_Results.csv', sep='\t',encoding='utf-8', index=False)
    print(df_experimentTracker.describe())
    
    return expExectr
 
def run_pcdnn_v1_experiments(dm, debug_mode = False):
    '''
    TODO: search for '#TODO:uncomment' in the 'experiment_executor/pcdnn_v1_experiment_executor.py' uncomment & comment out the necessary lines
    '''
    
    #dnnexperimentTrackingFields = ['Model','Dataset','Cpv Type','#Cpv',"ZmixExists",'MAE','TAE','MSE','TSE','#Pts','FitTime','PredTime','MAX-MAE','MAX-TAE','MAX-MSE','MAX-TSE','MIN-MAE','MIN-TAE','MIN-MSE','MIN-TSE']

    expExectr = PCDNNV1ExperimentExecutor()
    expExectr.debug_mode = debug_mode
    
    expExectr.setModelFactory(PCDNNV1ModelFactory())
    expExectr.executeExperiments(dm, "PCDNNV1",  pd.DataFrame())

    df_experimentTracker = expExectr.df_experimentTracker
    df_experimentTracker.to_csv('PCDNNV1_Experiment_Results.csv', sep='\t',encoding='utf-8', index=False)
    print(df_experimentTracker.describe())
    
    return expExectr
 
def run_pcdnn_v2_experiments(dm, n_models_override=None, n_epochs_override=None, use_dependants=False, debug_mode = False):
    '''
    TODO: search for '#TODO:uncomment' in the 'experiment_executor/pcdnn_v2_experiment_executor.py' uncomment & comment out the necessary lines
    '''
    #dnnexperimentTrackingFields = ['Model','Dataset','Cpv Type','#Cpv',"ZmixExists",'KernelConstraintExists','KernelRegularizerExists','ActivityRegularizerExists','MAE','TAE','MSE','TSE','#Pts','FitTime','PredTime','MAX-MAE','MAX-TAE','MAX-MSE','MAX-TSE','MIN-MAE','MIN-TAE','MIN-MSE','MIN-TSE']
    
    expExectr = PCDNNV2ExperimentExecutor()
    expExectr.debug_mode = debug_mode
    expExectr.use_dependants = use_dependants
    expExectr.n_models_override = n_models_override
    expExectr.n_epochs_override = n_epochs_override
 
    expExectr.setModelFactory(PCDNNV2ModelFactory())
    expExectr.executeExperiments(dm, "PCDNNV2", pd.DataFrame())
    
    df_experimentTracker = expExectr.df_experimentTracker 
    df_experimentTracker.to_csv('PCDNNV2_Experiment_Results.csv', sep='\t',encoding='utf-8', index=False)
    print(df_experimentTracker.describe())

    return expExectr

def run_model_experiments(dm, models=['PCDNN_V2'], debug_mode=False):
    # liberal inputs accepted!
    if type(models) is str: models = [models] # also accepts just 1 string
    supported_models = {'PCDNN_V2': run_pcdnn_v2_experiments, 'PCDNN_V1': run_pcdnn_v1_experiments, 'GP': run_gp_experiments, 'SIMPLE_DNN': run_simple_dnn_experiments}
    supported_models = {k.replace('_',''): v for k,v in supported_models.items()}
   
    expExectrs = []
    for model in models:
        model = model.upper().replace('_','')
        assert model in supported_models
        expExectrs.append(supported_models[model](dm, debug_mode=debug_mode))
    
    dm.save_PCA_data(fn='PCA_data.csv') # save updated PCA data
    compile_results()
    return expExectrs if len(expExectrs)>1 else expExectrs[0] 

def main(debug_mode=False):
    #Prepare the DataFrame that will be used downstream
    dp = DataPreparer()
    dp.createPCAs()
    dp.sparsePCAs()
    dp.zmixOrthogonalPCAs()
    df = dp.getDataframe()

    '''
    print(df[dp.pure_pca_dim_cols].describe().transpose())
    print(df[dp.sparse_pca_dim_cols].describe().transpose())
    print(df[dp.zmix_pca_dim_cols].describe().transpose())
    '''
    # currently passing dp eventually we want to abstract all the constants into 1 class
    dm = DataManager(df, dp)
 
    '''
    Run the PCDNN_v2 Experiments
    '''
    run_pcdnn_v2_experiments(dm, n_models_override=5, n_epochs_override=200, use_dependants=True, debug_mode = False)
	#run_model_experiments(dm, models='PCDNN_V2', debug_mode=debug_mode)
    '''
    Run the PCDNN_v1 Experiments
    '''
    #run_model_experiments(dm, models='PCDNN_V1', debug_mode=debug_mode)
    '''
    Run the Simple DNN Experiments
    '''
    #run_model_experiments(dm, models='SIMPLE_DNN', debug_mode=debug_mode)
    '''
    Run the GP Experiments 
    '''
    #run_model_experiments(dm, models='GP', debug_mode=debug_mode)


if __name__ == "__main__":
    main()    
