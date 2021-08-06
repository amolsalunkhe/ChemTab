# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 20:41:06 2021

@author: amol
"""
import pandas as pd
class ErrorManager:    
    def __init__(self):
        #print("Error Manager Instantiated")
        return 
    
    def computeError (self,Y_pred, Y_test):
        evaluation_df_1 = pd.DataFrame()

        evaluation_df_1['souener'] = Y_test.flatten()

        evaluation_df_1['souener_pred'] = Y_pred.flatten()

        evaluation_df_1['souener_pred_L1'] = evaluation_df_1['souener'] - evaluation_df_1['souener_pred'] 

        evaluation_df_1['souener_pred_L2'] = evaluation_df_1['souener_pred_L1'] * evaluation_df_1['souener_pred_L1']

        evaluation_df_1['souener_pred_L1Percent'] = ((evaluation_df_1['souener'] - evaluation_df_1['souener_pred'])/evaluation_df_1['souener']) 

        TotalAbsoluteError = evaluation_df_1['souener_pred_L1'].abs().sum()

        TotalSquaredError = evaluation_df_1['souener_pred_L2'].abs().sum()

        MeanAbsoluteError = evaluation_df_1['souener_pred_L1'].abs().sum()/evaluation_df_1['souener_pred_L1'].abs().count()

        MeanSquaredError = evaluation_df_1['souener_pred_L2'].abs().sum()/evaluation_df_1['souener_pred_L2'].abs().count()

        NumPoints = evaluation_df_1['souener_pred_L1Percent'].abs().count()

        MeanPercentageError = evaluation_df_1['souener_pred_L1Percent'].abs().sum()/NumPoints

        return [TotalAbsoluteError,TotalSquaredError,MeanAbsoluteError,MeanSquaredError,MeanPercentageError,NumPoints]
            
    def printError (self,err):
        TotalAbsoluteError = err[0]

        TotalSquaredError = err[1]

        MeanAbsoluteError = err[2]

        MeanSquaredError = err[3]

        MeanPercentageError = err[4]

        NumPoints = err[5]
        
        print ('Total Absolute Error: ', TotalAbsoluteError)
        print ('Mean Absolute Error: ', MeanAbsoluteError)
        print ('Mean Percentage Error: ', MeanPercentageError)
        print ('Total Squared Error: ', TotalSquaredError)
        print ('Mean Squared Error: ', MeanSquaredError)
        print ('Number of Points: ', NumPoints)
