
import time
import pandas as pd
from .error_manager import ErrorManager

class DNNExperimentExecutor:
    def __init__(self):
        self.dm = None
        self.modelType = None
        self.model = None
        self.df_experimentTracker = None
        self.fit_time = None
        self.pred_time = None
        self.err = None
        self.df_err = None 
        self.errManager = ErrorManager()
        self.modelFactory = None
        self.min_mae = 0
        
    def setModel(self,model):
        self.model = model
        
    def setModelFactory(self,modelFactory):
        self.modelFactory = modelFactory


    def executeSingleExperiment(self,experiment_num,noOfInputNeurons,dataSetMethod,dataType,inputType,ZmixPresent,noOfCpv):
        
        print('------------------ ' + 'executeSingleExperiments' + ' ------------------')
        

        #                           dataSetMethod,noOfCpvs, ipscaler, opscaler
        ipscaler = None
        opscaler = None
        self.dm.createTrainTestData(dataSetMethod,noOfCpv, ipscaler,  opscaler)

        self.modelFactory.experimentSettings = {"dataSetMethod": dataSetMethod,"ipscaler":ipscaler, "opscaler":opscaler, "noOfCpv": noOfCpv, "ZmixPresent": ZmixPresent }

        self.model = self.modelFactory.build_and_compile_model(noOfInputNeurons)
        
        self.model.summary()

        self.fitModelAndCalcErr(self.dm.X_train, self.dm.Y_train, self.dm.X_test, self.dm.Y_test)

        #['Model','Dataset','Cpv Type','#Cpv',"ZmixExists",'MAE','TAE','MSE','TSE','#Pts','FitTime','PredTime','MAX-MAE','MAX-TAE','MAX-MSE','MAX-TSE','MIN-MAE','MIN-TAE','MIN-MSE','MIN-TSE']

        # log experiment results
        distribution_summary_stats = lambda error_df, target_key: {target_key + '-MIN': error_df[target_key].min(),
                                                                   target_key: error_df[target_key].mean(),
                                                                   target_key + '-MAX': error_df[target_key].max()}

        experimentResults = {'Model': self.modelType, 'Dataset':dataType, 'Cpv Type':inputType, '#Cpv': noOfCpv, "ZmixExists": ZmixPresent, 'FitTime': self.fit_time, 'PredTime': self.pred_time}
        err_names = ['MAE', 'TAE', 'MSE', 'TSE', 'MRE', 'TRE']
        for name in err_names:
            experimentResults.update(distribution_summary_stats(self.df_err, name))

        #experimentResults = [self.modelType, dataType,inputType,str(noOfCpv), ZmixPresent,str(self.df_err['MAE'].mean()),str(self.df_err['TAE'].mean()),str(self.df_err['MSE'].mean()),str(self.df_err['TSE'].mean()),str(self.df_err['#Pts'].mean()),str(self.fit_time),str(self.pred_time),str(self.df_err['MAE'].max()),str(self.df_err['TAE'].max()),str(self.df_err['MSE'].max()),str(self.df_err['TSE'].max()),str(self.df_err['MAE'].min()),str(self.df_err['TAE'].min()),str(self.df_err['MSE'].min()),str(self.df_err['TSE'].min())]

        self.df_experimentTracker.loc[len(self.df_experimentTracker)] = experimentResults        

        printStr = "self.modelType: "+ self.modelType+ " dataType: "  + dataType+ " inputType:"+inputType+ " noOfCpv:"+str(noOfCpv)+ " ZmixPresent:" + ZmixPresent + " MAE:" +str(self.df_err['MAE'].min())

        print(printStr)

        printStr = "\t"

        printStr = printStr.join(experimentResults)

    def executeExperiments(self,dataManager, modelType, df_experimentTracker):
        self.dm = dataManager
        self.modelType = modelType
        self.df_experimentTracker = df_experimentTracker
        experiment_num = 0
        #Experiments  
        
        if self.debug_mode:
            #TODO:comment
            dataTypes = ["randomequalflamesplit"]
            inputTypes = ["ZmixCpv"]
        else:
            #TODO:uncomment
            dataTypes = ["frameworkincludedtrainexcludedtest", "randomequalflamesplit"]#, "randomequaltraintestsplit"] #for production -- uncomment this
            inputTypes = ["ZmixCpv","ZmixPCA","SparsePCA","PurePCA","ZmixAndPurePCA","ZmixAndSparsePCA","ZmixAllSpecies","AllSpecies"]

        for dataType in dataTypes:
            print('=================== ' + dataType + ' ===================')
            
            for inputType in inputTypes:
                
                print('------------------ ' + inputType + ' ------------------')
                    
                #ZmixCpv_randomequaltraintestsplit
                dataSetMethod = inputType + '_' + dataType
                
                self.modelFactory.setDataSetMethod(dataSetMethod)
                
                noOfNeurons = 0
                #ZmixAnd & ZmixAll
                if inputType.find('ZmixA') != -1:
                    noOfNeurons = noOfNeurons + 1

                if inputType.find('Zmix') != -1:
                    ZmixPresent = 'Y'
                else:
                    ZmixPresent = 'N'
                    
                if inputType.find('PCA') != -1:


                    m = 2 if self.debug_mode else 6
                    noOfCpvs = [item for item in range(1, m)]
                    
                    for noOfCpv in noOfCpvs:
                        noOfNeurons = noOfNeurons + 1                        
                        experiment_num = experiment_num + 1
                        self.executeSingleExperiment(experiment_num,noOfNeurons,dataSetMethod,dataType,inputType,ZmixPresent,noOfCpv)
                else:

                    if inputType.find('ZmixCpv') != -1:
                        noOfCpv = 1
                        noOfNeurons = 2
                    else:
                        noOfCpv = 53
                        noOfNeurons = noOfNeurons + noOfCpv
                    print('------------------ ' + str(noOfNeurons) + ' ------------------')
                    experiment_num = experiment_num + 1
                    self.executeSingleExperiment(experiment_num,noOfNeurons,dataSetMethod,dataType,inputType,ZmixPresent,noOfCpv)
        

        
    def fitModelAndCalcErr(self,X_train, Y_train, X_test, Y_test):

        fit_times = []
        
        pred_times = []
        
        errs = []
        
        #temp = 0
        n = 2 if self.debug_mode else 11

        for itr in range(1,n):
            print(f'training model: {itr}')
            t = time.process_time()

            self.model.fit(X_train, {"prediction":Y_train},validation_split=0.2,verbose=0,epochs=100)

            fit_times.append(time.process_time() - t)
        
            t = time.process_time()

            Y_pred = self.model.predict(X_test)
            
            #sns.residplot(Y_pred.flatten(), getResiduals(Y_test,Y_pred))

            pred_times.append(time.process_time() - t)
            
            curr_errs = self.errManager.computeError (Y_pred, Y_test)
                
            if (len(errs) == 0) or ((len(errs) > 0) and (curr_errs['MAE'] < self.min_mae)) :
                self.min_mae = curr_errs['MAE']#MAE
                self.modelFactory.saveCurrModelAsBestModel()
                #temp = temp + 1
                
            errs.append(curr_errs)
        #print ("Model saved #: " + str(temp)) 
        #print (errs)       
        self.fit_time = sum(fit_times)/len(fit_times)
        
        self.pred_time = sum(pred_times)/len(pred_times)
        
        #computeAndPrintError(Y_pred, Y_test)

        self.df_err = pd.DataFrame(errs, columns = ['TAE', 'TSE', 'MAE', 'MSE', 'MAPE', '#Pts'])
        
        return
