import time
import pandas as pd
class GPExperimentExecutor:
    def __init__(self):
        self.dm = None
        self.modelType = None
        self.model = None
        self.df_experimentTracker = None
        self.fit_time = None
        self.pred_time = None
        self.err = None
     
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

        experimentResults = [self.modelType, dataType,inputType,str(noOfCpv), ZmixPresent,str(self.err[2]),str(self.err[0]),str(self.err[3]),str(self.err[1]),str(self.err[5]),str(self.fit_time),str(self.pred_time)]

        printStr = "self.modelType: "+ self.modelType+ " dataType: "  + dataType+ " inputType:"+inputType+ " noOfCpv:"+str(noOfCpv)+ " ZmixPresent:" + ZmixPresent + " MAE:" +str(self.err[2])

        print(printStr)
        
    def executeExperiments(self,dataManager, modelType, df_experimentTracker):
        self.dm = dataManager
        self.modelType = modelType
        self.df_experimentTracker = df_experimentTracker
        
        #Experiments  
        
		#TODO:uncomment
        #dataTypes = ["randomequaltraintestsplit","frameworkincludedtrainexcludedtest"] #for production -- uncomment this
        #inputTypes = ["ZmixCpv","ZmixPCA","SparsePCA","PurePCA","ZmixAndPurePCA","ZmixAndSparsePCA","ZmixAllSpecies","AllSpecies"] #for production -- uncomment this
        
		#TODO:comment        
		dataTypes = ["frameworkincludedtrainexcludedtest"] #for testing -- comment this 
        inputTypes = ["ZmixCpv"] #for testing -- comment this
        
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
                    
                    noOfCpvs = [item for item in range(1, 6)]
                    
                    for noOfCpv in noOfCpvs:
                        #                           dataSetMethod,noOfCpvs, ipscaler, opscaler
                        self.dm.createTrainTestData(dataSetMethod,noOfCpv, None, None)
                        
                        self.fitModelAndCalcErr(self.dm.X_train, self.dm.Y_train, self.dm.X_test, self.dm.Y_test)
                                                
                        experimentResults = [self.modelType, dataType,inputType,str(noOfCpv), ZmixPresent,str(self.err[2]),str(self.err[0]),str(self.err[3]),str(self.err[1]),str(self.err[5]),str(self.fit_time),str(self.pred_time)]

                        self.df_experimentTracker.loc[len(self.df_experimentTracker)] = experimentResults        

                        printStr = "self.modelType: "+ self.modelType+ " dataType: "  + dataType+ " inputType:"+inputType+ " noOfCpv:"+str(noOfCpv)+ " ZmixPresent:" + ZmixPresent + " MAE:" +str(self.err[2])

                        print(printStr)
                        
                        
                else:
                   
                    if inputType.find('ZmixCpv') != -1:
                        noOfCpv = 1
                    else:
                        noOfCpv = 53
                        
                    #                           dataSetMethod,noOfCpvs, ipscaler, opscaler
                    self.dm.createTrainTestData(dataSetMethod,noOfCpv, None, None)

                    self.fitModelAndCalcErr(self.dm.X_train, self.dm.Y_train, self.dm.X_test, self.dm.Y_test)    

                    experimentResults = [self.modelType, dataType,inputType,str(noOfCpv), ZmixPresent,str(self.err[2]),str(self.err[0]),str(self.err[3]),str(self.err[1]),str(self.err[5]),str(self.fit_time),str(self.pred_time)]
                        
                    self.df_experimentTracker.loc[len(self.df_experimentTracker)] = experimentResults        

                    printStr = "self.modelType: "+ self.modelType+ " dataType: "  + dataType+ " inputType:"+inputType+ " noOfCpv:"+str(noOfCpv)+ " ZmixPresent:" + ZmixPresent + " MAE:" +str(self.err[2])

                    print(printStr)

                    printStr = "\t"

                    printStr = printStr.join(experimentResults)

        
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

        
        
    def fitModelAndCalcErr(self,X_train, Y_train, X_test, Y_test):

        t = time.process_time()

        self.model.fit(X_train, Y_train)

        self.fit_time = time.process_time() - t

        t = time.process_time()

        Y_pred = self.model.predict(X_test, return_std=False)

        self.pred_time = time.process_time() - t
        
        #computeAndPrintError(Y_pred, Y_test)

        self.err = self.computeError (Y_pred, Y_test)
        
        return 