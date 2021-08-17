# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 17:50:06 2021

@author: amol
"""


from data.pre_processing import DataPreparer 
from data.train_test_manager import DataManager 
from experiment_executor.gp_experiment_executor import GPExperimentExecutor
from experiment_executor.simple_dnn_experiment_executor import DNNExperimentExecutor
from experiment_executor.pcdnn_v1_experiment_executor import PCDNNV1ExperimentExecutor
from experiment_executor.pcdnn_v2_experiment_executor import PCDNNV2ExperimentExecutor

from models.gpmodel import GPModel
from models.simplednn_model_factory import SimpleDNNModelFactory
from models.pcdnnv1_model_factory import PCDNNV1ModelFactory
from models.pcdnnv2_model_factory import PCDNNV2ModelFactory
import pandas as pd

def run_gp_experiments(dm):
    '''
    TODO: search for '#TODO:uncomment' in the 'experiment_executor/gp_experiment_executor.py' uncomment & comment out the necessary lines
    '''
    gp = GPModel()
    experimentTrackingFields = ['Model','Dataset','Cpv Type','#Cpv',"ZmixExists",'MAE','TAE','MSE','TSE','#Pts','FitTime','PredTime']
    df_experimentTracker = pd.DataFrame(columns=experimentTrackingFields)
    
    expExectr = GPExperimentExecutor()
    expExectr.setModel(gp.getModel("Matern"))
    expExectr.executeExperiments(dm, "GP_Matern", df_experimentTracker)
    
    expExectr = GPExperimentExecutor()
    expExectr.setModel(gp.getModel("RationalQuadratic"))
    expExectr.executeExperiments(dm, "GP_RationalQuadratic", df_experimentTracker)
    
    expExectr = GPExperimentExecutor()
    expExectr.setModel(gp.getModel("Matern_RationalQuadratic"))
    expExectr.executeExperiments(dm, "GP_Matern_RationalQuadratic", df_experimentTracker)

    df_experimentTracker.to_csv('GP_Experiment_Results.csv', sep='\t',encoding='utf-8', index=False)
    
    print(df_experimentTracker.describe())

def run_simple_dnn_experiments(dm):
    '''
    TODO: search for '#TODO:uncomment' in the 'experiment_executor/simple_dnn_experiment_executor.py' uncomment & comment out the necessary lines
    '''
    dnnexperimentTrackingFields = ['Model','Dataset','Cpv Type','#Cpv',"ZmixExists",'MAE','TAE','MSE','TSE','#Pts','FitTime','PredTime','MAX-MAE','MAX-TAE','MAX-MSE','MAX-TSE','MIN-MAE','MIN-TAE','MIN-MSE','MIN-TSE']

    df_dnnexperimentTracker = pd.DataFrame(columns=dnnexperimentTrackingFields)

    expExectr = DNNExperimentExecutor()

    expExectr.setModelFactory(SimpleDNNModelFactory())

    expExectr.executeExperiments(dm, "Simple_DNN", df_dnnexperimentTracker)

    df_dnnexperimentTracker.to_csv('SimpleDNN_Experiment_Results.csv', sep='\t',encoding='utf-8', index=False)
    
    print(df_dnnexperimentTracker.describe())
 
def run_pcdnn_v1_experiments(dm):
    '''
    TODO: search for '#TODO:uncomment' in the 'experiment_executor/pcdnn_v1_experiment_executor.py' uncomment & comment out the necessary lines
    '''
    dnnexperimentTrackingFields = ['Model','Dataset','Cpv Type','#Cpv',"ZmixExists",'MAE','TAE','MSE','TSE','#Pts','FitTime','PredTime','MAX-MAE','MAX-TAE','MAX-MSE','MAX-TSE','MIN-MAE','MIN-TAE','MIN-MSE','MIN-TSE']

    df_pcdnnexperimentTracker = pd.DataFrame(columns=dnnexperimentTrackingFields)

    expExectr = PCDNNV1ExperimentExecutor()
    
    expExectr.setModelFactory(PCDNNV1ModelFactory())
    
    expExectr.executeExperiments(dm, "PCDNNV1", df_pcdnnexperimentTracker)

    df_pcdnnexperimentTracker.to_csv('PCDNNV1_Experiment_Results.csv', sep='\t',encoding='utf-8', index=False)
    
    print(df_pcdnnexperimentTracker.describe())
 
def run_pcdnn_v2_experiments(dm):
    '''
    TODO: search for '#TODO:uncomment' in the 'experiment_executor/pcdnn_v2_experiment_executor.py' uncomment & comment out the necessary lines
    '''
    
    dnnexperimentTrackingFields = ['Model','Dataset','Cpv Type','#Cpv',"ZmixExists",'KernelConstraintExists','KernelRegularizerExists','ActivityRegularizerExists','MAE','TAE','MSE','TSE','#Pts','FitTime','PredTime','MAX-MAE','MAX-TAE','MAX-MSE','MAX-TSE','MIN-MAE','MIN-TAE','MIN-MSE','MIN-TSE']
    
    df_pcdnnexperimentTracker = pd.DataFrame(columns=dnnexperimentTrackingFields)

    expExectr = PCDNNV2ExperimentExecutor()
    
    expExectr.setModelFactory(PCDNNV2ModelFactory())
    
    expExectr.executeExperiments(dm, "PCDNNV2", df_pcdnnexperimentTracker)

    df_pcdnnexperimentTracker.to_csv('PCDNNV2_Experiment_Results.csv', sep='\t',encoding='utf-8', index=False)

    bestModel, experimentSettings = expExectr.modelFactory.openBestModel()

    linearAutoEncoder = expExectr.modelFactory.getLinearEncoder()     
    
    
    dm.createTrainTestData(experimentSettings.get('dataSetMethod'), experimentSettings.get('noOfCpv'), experimentSettings.get('inputScaler'), experimentSettings.get('outputScaler'))
    
    #X_train, X_test, Y_train, Y_test, rom_train, rom_test, zmix_train, zmix_test = dm.getTrainTestData() 
    
    X,Y,rom,zmix = dm.getAllData()
    
    pcdnnv2LinearEmbedding = linearAutoEncoder.predict(X)
    
    print (pcdnnv2LinearEmbedding)

    #Get the Regressor
    regressor = expExectr.modelFactory.getRegressor()
        
    reg_predicts = regressor.predict([zmix,pcdnnv2LinearEmbedding])#predict({'zmix':zmix,'linear_embedding':pcdnnv2LinearEmbedding})
    print('Regressor Predicts')
    print(reg_predicts)
    
    
    bestModelPredicts = bestModel.predict({'zmix':zmix,'species_input':X})
    print('Best Model Predicts')
    print(bestModelPredicts)
    
    evaluation_df_1 = pd.DataFrame()

    evaluation_df_1['best_model'] = bestModelPredicts.flatten()

    evaluation_df_1['regressor'] = reg_predicts.flatten()
    
    print(evaluation_df_1.describe())
    
    #df_pcdnnv2LinearEmbedding.to_csv('PCDNNV2_Linear_Embeddings.csv', sep='\t',encoding='utf-8', index=False)

    '''
    TODO: Add the ErrorAnalysis | Xpos & flameid
          Add the Explainability
          
    '''
    
    print(df_pcdnnexperimentTracker.describe())
 
    '''
    print(dataSetMethod)
    
    print("-----------------------------------------------")
    
    bestModel.summary()
    '''
def main():
    #Prepare the DataFrame that will be used downstream
    dp = DataPreparer()
    dp.createPCAs()
    dp.sparsePCAs()
    dp.zmixOrthogonalPCAs()
    df = dp.getDataframe()

    df.to_csv('PCA_data.csv', index=False)
   
    '''
    print(df[dp.pure_pca_dim_cols].describe().transpose())
    print(df[dp.sparse_pca_dim_cols].describe().transpose())
    print(df[dp.zmix_pca_dim_cols].describe().transpose())
    '''
    # currently passing dp eventually we want to abstract all the constants into 1 class
    dm = DataManager(df, dp)
    
    '''
    1. Run the GP Experiments
     
    '''
    #run_gp_experiments(dm)
    
    '''
    2. Run the Simple DNN Experiments
    '''
    #run_simple_dnn_experiments(dm)
        
    '''
    3. Run the PCDNN_v1 Experiments
    '''
    #run_pcdnn_v1_experiments(dm)
    
    '''
    4. Run the PCDNN_v2 Experiments
    '''
    run_pcdnn_v2_experiments(dm)
    
    
if __name__ == "__main__":
    main()    
