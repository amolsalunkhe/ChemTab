import time
import pandas as pd
from .error_manager import ErrorManager
from copy import deepcopy


class GPExperimentExecutor:
    def __init__(self):
        self.dm = None
        self.modelType = None
        self.model = None
        self.df_experimentTracker = None
        self.errManager = ErrorManager()     
        self.fit_time = None
        self.pred_time = None
        self.err = None
        self.debug_mode = False


    def setModel(self,model):
        self.model = model

    def executeExperiment(self,dataManager, modelType, dataType="randomequaltraintestsplit",inputType="ZmixPCA",noOfCpv=5):
        self.dm = dataManager
        
        self.modelType = modelType
        
        dataSetMethod = inputType + '_' + dataType
        
        self.dm.createTrainTestData(dataSetMethod,noOfCpv, None, None)
        
        self.fitModelAndCalcErr(self.dm.X_train, self.dm.Y_train, self.dm.X_test, self.dm.Y_test)
        
        if inputType.find('Zmix') != -1:
            ZmixPresent = 'Y'
        else:
            ZmixPresent = 'N'

        experimentResults = deepcopy(self.err)
        experimentResults.update({'Model': self.modelType, 'Dataset': dataType, 'Cpv Type': inputType, '#Cpv': str(noOfCpv),
                                  'ZmixExists': ZmixPresent, 'FitTime': self.fit_time, 'PredTime': self.pred_time})
        #experimentResults = [self.modelType, dataType,inputType,str(noOfCpv), ZmixPresent,str(self.err[2]),str(self.err[0]),str(self.err[3]),str(self.err[1]),str(self.err[5]),str(self.fit_time),str(self.pred_time)]

        printStr = "self.modelType: "+ self.modelType+ " dataType: "  + dataType+ " inputType:"+inputType+ " noOfCpv:"+str(noOfCpv)+ " ZmixPresent:" + ZmixPresent + " MAE:" +str(self.err['MAE'])

        print(printStr)
        
    def executeExperiments(self,dataManager, modelType, df_experimentTracker):
        self.dm = dataManager
        self.modelType = modelType
        self.df_experimentTracker = df_experimentTracker
        
        #Experiments  
        
        if self.debug_mode:
            #TODO:comment 
            dataTypes = ["randomequalflamesplit"] #for testing -- comment this 
            inputTypes = ["ZmixCpv"] #for testing -- comment this
        else:
            #TODO:uncomment
            dataTypes = ["frameworkincludedtrainexcludedtest", "randomequalflamesplit"]#, "randomequaltraintestsplit"] #for production -- uncomment this
            inputTypes = ["ZmixCpv","ZmixPCA","SparsePCA","PurePCA","ZmixAndPurePCA","ZmixAndSparsePCA","ZmixAllSpecies","AllSpecies"] #for production -- uncomment this


        for dataType in dataTypes:
            
            for inputType in inputTypes:
                #ZmixCpv_randomequaltraintestsplit
                #
                dataSetMethod = inputType + '_' + dataType
                        
                if inputType.find('Zmix') != -1:
                    ZmixPresent = 'Y'
                else:
                    ZmixPresent = 'N'
                    
                if inputType.find('PCA') != -1:
                    m = 3 if self.debug_mode else 6 
                    noOfCpvs = [item for item in range(2, m)]
                    
                    for noOfCpv in noOfCpvs:
                        #                           dataSetMethod,noOfCpvs, ipscaler, opscaler
                        self.dm.createTrainTestData(dataSetMethod,noOfCpv, None, None)
                        
                        self.fitModelAndCalcErr(self.dm.X_train, self.dm.Y_train, self.dm.X_test, self.dm.Y_test)
                                                
                        experimentResults = deepcopy(self.err)
                        experimentResults.update({'Model': self.modelType, 'Dataset': dataType, 'Cpv Type': inputType, '#Cpv': str(noOfCpv),
                                                  'ZmixExists': ZmixPresent, 'FitTime': self.fit_time, 'PredTime': self.pred_time})
                        #[self.modelType, dataType,inputType,str(noOfCpv), ZmixPresent,str(self.err[2]),str(self.err[0]),str(self.err[3]),str(self.err[1]),str(self.err[5]),str(self.fit_time),str(self.pred_time)]

                        self.df_experimentTracker.loc[len(self.df_experimentTracker)] = experimentResults        

                        printStr = "self.modelType: "+ self.modelType+ " dataType: "  + dataType+ " inputType:"+inputType+ " noOfCpv:"+str(noOfCpv)+ " ZmixPresent:" + ZmixPresent + " MAE:" +str(self.err['MAE'])
                        print(printStr)
                else:
                   
                    if inputType.find('ZmixCpv') != -1:
                        noOfCpv = 1
                    else:
                        noOfCpv = 53
                        
                    #                           dataSetMethod,noOfCpvs, ipscaler, opscaler
                    self.dm.createTrainTestData(dataSetMethod,noOfCpv, None, None)

                    self.fitModelAndCalcErr(self.dm.X_train, self.dm.Y_train, self.dm.X_test, self.dm.Y_test)    

                    experimentResults = deepcopy(self.err)
                    experimentResults.update({'Model': self.modelType, 'Dataset': dataType, 'Cpv Type': inputType, '#Cpv': str(noOfCpv),
                                              'ZmixExists': ZmixPresent, 'FitTime': self.fit_time, 'PredTime': self.pred_time})
                    #experimentResults = [self.modelType, dataType,inputType,str(noOfCpv), ZmixPresent,str(self.err[2]),str(self.err[0]),str(self.err[3]),str(self.err[1]),str(self.err[5]),str(self.fit_time),str(self.pred_time)]
                        
                    self.df_experimentTracker.loc[len(self.df_experimentTracker)] = experimentResults        

                    printStr = "self.modelType: "+ self.modelType+ " dataType: "  + dataType+ " inputType:"+inputType+ " noOfCpv:"+str(noOfCpv)+ " ZmixPresent:" + ZmixPresent + " MAE:" +str(self.err['MAE'])

                    print(printStr)

                    printStr = "\t"

                    printStr = printStr.join(experimentResults)
        
    def printError (self,err):
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

        
    def fitModelAndCalcErr(self,X_train, Y_train, X_test, Y_test):

        print(f'training model')

        t = time.process_time()

        self.model.fit(X_train, Y_train)

        self.fit_time = time.process_time() - t

        t = time.process_time()

        Y_pred = self.model.predict(X_test, return_std=False)

        self.pred_time = time.process_time() - t
        
        #computeAndPrintError(Y_pred, Y_test)

        self.err = self.errManager.computeError (Y_pred, Y_test)
        
        return 
