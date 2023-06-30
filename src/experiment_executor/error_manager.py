# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 20:41:06 2021

@author: amol
"""
import pandas as pd

'''
# No, nvm don't use me!
# TODO: use me!!
def computeErrorAcrossDependents(Y_pred_df, Y_test_df):
    err_data = []
    for dependent in dependents:
        Y_pred = Y_pred_df[dependent].values
        Y_test = Y_test_df[dependent].values
        err = computeError (Y_pred, Y_test)
        err.insert(0,dependent)
        err_data.append(err)
    err_df = pd.DataFrame(data=err_data, columns=["Dependent","Mean ABS","Total ABS","Mean Pct Err","Mean SSE","Total SSE", "#Pts" ])
    err_df.drop(columns=["Mean Pct Err","Mean SSE","Total SSE"], inplace=True)
    return err_df
'''

class ErrorManager:    
    def __init__(self):
        #print("Error Manager Instantiated")
        self.souener_index = 0 # should always be 0, still it is set manually later 
   
    #def set_souener_index(self, dm):
    #    self.souener_index = dm.output_data_cols.index('souener')
    #    assert self.souener_index == 0 # somewhat rediculous sanity check, since it is not dynamic
 
    def computeError (self, Y_pred, Y_test):
        # select souener only
        #Y_pred = Y_pred[:, self.souener_index]
        #Y_test = Y_test[:, self.souener_index]

        evaluation_df_1 = pd.DataFrame()

        evaluation_df_1['souener'] = Y_test.squeeze()

        evaluation_df_1['souener_pred'] = Y_pred.squeeze()

        evaluation_df_1['souener_pred_L1'] = (evaluation_df_1['souener'] - evaluation_df_1['souener_pred']).abs() 

        evaluation_df_1['souener_pred_L2'] = evaluation_df_1['souener_pred_L1'] * evaluation_df_1['souener_pred_L1']

        evaluation_df_1['souener_pred_L1Percent'] = ((evaluation_df_1['souener'] - evaluation_df_1['souener_pred'])/evaluation_df_1['souener']) 

        TotalAbsoluteError = evaluation_df_1['souener_pred_L1'].abs().sum()

        TotalSquaredError = evaluation_df_1['souener_pred_L2'].abs().sum()

        MeanAbsoluteError = evaluation_df_1['souener_pred_L1'].abs().sum()/evaluation_df_1['souener_pred_L1'].abs().count()

        MeanSquaredError = evaluation_df_1['souener_pred_L2'].abs().sum()/evaluation_df_1['souener_pred_L2'].abs().count()

        MeanRelativeError = (evaluation_df_1['souener_pred_L1']/evaluation_df_1['souener']).abs().sum()/evaluation_df_1['souener_pred_L1'].abs().count()
        #TotalRelativeError = (evaluation_df_1['souener_pred_L1']/evaluation_df_1['souener']).abs().sum()/evaluation_df_1['souener_pred_L1'].abs().count()

        NumPoints = evaluation_df_1['souener_pred_L1Percent'].abs().count()

        MeanPercentageError = evaluation_df_1['souener_pred_L1Percent'].abs().sum()/NumPoints


        columns = ['MAE', 'MSE', 'MAPE', 'MRE', '#Pts']
        error_row = [MeanAbsoluteError, MeanSquaredError, MeanPercentageError, MeanRelativeError, NumPoints]
        return {k: v for k,v in zip(columns, error_row)}

    def getExperimentErrorResults(self, err_df):
        #['Model','Dataset','Cpv Type','#Cpv',"ZmixExists",'MAE','TAE','MSE','TSE','#Pts','FitTime','PredTime','MAX-MAE','MAX-TAE','MAX-MSE','MAX-TSE','MIN-MAE','MIN-TAE','MIN-MSE','MIN-TSE']

        #experimentResults = {'Model': self.modelType, 'Dataset':dataType, 'Cpv Type':inputType, '#Cpv':noOfCpv,
        #                     'ZmixExists': ZmixPresent, '#Pts': self.df_err['#Pts'].mean(), 'FitTime': self.fit_time, 'PredTime': self.pred_time,
        #                     'KernelConstraintExists': kernel_constraint, 'KernelRegularizerExists': kernel_regularizer,'ActivityRegularizerExists': activity_regularizer,
        #                     'OPScaler': opscaler} 
        distribution_summary_stats = lambda error_df, target_key: {'MIN-' + target_key: error_df[target_key].min(),
                                                                   target_key: error_df[target_key].mean(),
                                                                   'MAX-' + target_key: error_df[target_key].max()}
        # TODO: add R^2 error in here!
        err_names = ['MAE', 'TAE', 'MSE', 'TSE', 'MRE', 'TRE']
        errorResults = {'#Pts': err_df['#Pts'].mean()}
        for name in err_names:
            errorResults.update(distribution_summary_stats(err_df, name))
        return errorResults
 
    def printError (self,err):
        if type(err) is pd.DataFrame:
            print(err.describe())
        TotalAbsoluteError = err['TAE']

        TotalSquaredError = err['TSE']

        MeanAbsoluteError = err['MAE']

        MeanSquaredError = err['MSE']

        MeanPercentageError = err['MAPE']

        NumPoints = err['#Pts']
        
        print ('Total Absolute Error: ', TotalAbsoluteError)
        print ('Mean Absolute Error: ', MeanAbsoluteError)
        print ('Mean Percentage Error: ', MeanPercentageError)
        print ('Total Squared Error: ', TotalSquaredError)
        print ('Mean Squared Error: ', MeanSquaredError)
        print ('Number of Points: ', NumPoints)
