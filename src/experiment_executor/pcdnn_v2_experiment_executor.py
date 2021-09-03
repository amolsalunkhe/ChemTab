# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 09:25:55 2021

@author: amol
"""
import time
import pandas as pd
import matplotlib.pyplot as plt

from .error_manager import ErrorManager

class PCDNNV2ExperimentExecutor:
    def __init__(self):
        self.dm = None
        self.modelType = None
        self.model = None
        self.df_experimentTracker = None
        self.fit_time = None
        self.pred_time = None
        self.err = None
        self.df_err = None 
        self.predicitions = None
        self.errManager = ErrorManager()
        self.modelFactory = None
        self.min_mae = 0
        
    def setModel(self,model):
        self.model = model
    
    def setModelFactory(self,modelFactory):
        self.modelFactory = modelFactory    
    
    def getPredicitons(self):
        return self.predicitions

    
    def executeSingleExperiment(self,noOfInputNeurons,dataSetMethod,dataType,inputType,ZmixPresent,noOfCpv,concatenateZmix,kernel_constraint,kernel_regularizer,activity_regularizer):

        
        print("--------------------self.build_and_compile_pcdnn_v2_model----------------------")
        
        self.model = self.modelFactory.build_and_compile_model(noOfInputNeurons,noOfCpv,concatenateZmix,kernel_constraint,kernel_regularizer,activity_regularizer)
            
        self.model.summary()

        #                           dataSetMethod,noOfCpvs, ipscaler, opscaler
        ipscaler = "MinMaxScaler"
        opscaler = "MinMaxScaler"
        self.dm.createTrainTestData(dataSetMethod,noOfCpv, ipscaler,  opscaler)

        self.modelFactory.experimentSettings = {"dataSetMethod": dataSetMethod,"ipscaler":ipscaler, "opscaler":opscaler, "noOfCpv": noOfCpv, "ZmixPresent": ZmixPresent, "concatenateZmix": concatenateZmix,"kernel_constraint":kernel_constraint,"kernel_regularizer":kernel_regularizer,"activity_regularizer":activity_regularizer }

        X_train, X_test, Y_train, Y_test, rom_train, rom_test, zmix_train, zmix_test = self.dm.getTrainTestData() 
        
        self.fitModelAndCalcErr(X_train, Y_train, X_test, Y_test,None, None, zmix_train, zmix_test, self.dm.outputScaler, concatenateZmix,kernel_constraint,kernel_regularizer,activity_regularizer)


        #['Model','Dataset','Cpv Type','#Cpv',"ZmixExists",'MAE','TAE','MSE','TSE','#Pts','FitTime','PredTime','MAX-MAE','MAX-TAE','MAX-MSE','MAX-TSE','MIN-MAE','MIN-TAE','MIN-MSE','MIN-TSE']

        distribution_summary_stats = lambda error_df, target_key: {target_key + '-MIN': error_df[target_key].min(), 
                                                                   target_key: error_df[target_key].mean(), 
                                                                   target_key + '-MAX': error_df[target_key].max()}
        
        experimentResults = {'Model': self.modelType, 'Dataset':dataType, 'Cpv Type':inputType, '#Cpv':noOfCpv, 'KernelConstraintExists': kernel_constraint,
                             'KernelRegularizerExists': kernel_regularizer,'ActivityRegularizerExists': activity_regularizer,
                             "ZmixExists": ZmixPresent, '#Pts': self.df_err['#Pts'].mean(), 'FitTime': self.fit_time, 'PredTime': self.pred_time}
        err_names = ['MAE', 'TAE', 'MSE', 'TSE', 'MRE', 'TRE']
        for name in err_names:
            experimentResults.update(distribution_summary_stats(self.df_err, name))

        self.df_experimentTracker.loc[len(self.df_experimentTracker)] = experimentResults        

        printStr = "self.modelType: "+ self.modelType+ " dataType: "  + dataType+ " inputType:"+inputType+ " noOfCpv:"+str(noOfCpv)+ " ZmixPresent:" + ZmixPresent + " MAE:" +str(self.df_err['MAE'].min())

        print(printStr)

        printStr = "\t"

        printStr = printStr.join(experimentResults)



    def fitModelAndCalcErr(self,X_train, Y_train, X_test, Y_test, rom_train = None, rom_test = None, zmix_train = None, zmix_test = None, Y_scaler = None, concatenateZmix = 'N',kernel_constraint = 'Y',kernel_regularizer = 'Y',activity_regularizer = 'Y'):

        fit_times = []
        
        pred_times = []
        
        errs = []
        
        n = 3 if self.debug_mode else 11
        for itr in range(1,n):    
        
            print(f'training model: {itr}')
            t = time.process_time()

            if concatenateZmix == 'Y':
                history = self.model.fit({"species_input":X_train, "zmix":zmix_train}, {"prediction":Y_train},validation_split=0.2,verbose=0,epochs=100)
            else:
                history = self.model.fit({"species_input":X_train}, {"prediction":Y_train},validation_split=0.2,verbose=0,epochs=100)
            
            #self.plot_loss_physics_and_regression(history)
            
            fit_times.append(time.process_time() - t)
        
            t = time.process_time()

            if concatenateZmix == 'Y':
                predictions = self.model.predict({"species_input":X_test, "zmix":zmix_test})
            else:
                predictions = self.model.predict({"species_input":X_test})
                
            pred_times.append(time.process_time() - t)
            
            self.predicitions = predictions
            
            Y_pred = predictions
            

            if Y_scaler is not None:
                Y_pred = Y_scaler.inverse_transform(Y_pred)
                
                
            #sns.residplot(Y_pred.flatten(), getResiduals(Y_test,Y_pred))

            curr_errs = self.errManager.computeError (Y_pred, Y_test)
                
            if (len(errs) == 0) or ((len(errs) > 0) and (curr_errs['MAE'] < self.min_mae)) :
                self.min_mae = curr_errs['MAE']#MAE
                self.modelFactory.saveCurrModelAsBestModel()
                    
            errs.append(curr_errs)
        
        
        
        
        self.fit_time = sum(fit_times)/len(fit_times)
        
        self.pred_time = sum(pred_times)/len(pred_times)
        
        #computeAndPrintError(Y_pred, Y_test)

        self.df_err = pd.DataFrame(errs)
        
        return  

    def executeExperiments(self,dataManager, modelType, df_experimentTracker):
        self.dm = dataManager
        self.modelType = modelType
        self.df_experimentTracker = df_experimentTracker
        
        #Experiments  
        if self.debug_mode:
            #TODO:comment
            dataTypes = ["randomequalflamesplit"]
            inputTypes = ["AllSpeciesAndZmix"]    
        else:
            #TODO:uncomment
            dataTypes = ["frameworkincludedtrainexcludedtest", "randomequalflamesplit"]#, "randomequaltraintestsplit"]
            inputTypes = ["AllSpecies","AllSpeciesAndZmix"]
        
        concatenateZmix = 'N'
        
        if not self.debug_mode:
            #TODO:uncomment
            kernel_constraints = ['Y','N']
            kernel_regularizers = ['Y','N']
            activity_regularizers = ['Y','N']        
        else:
            #TODO:comment
            kernel_constraints = ['Y']
            kernel_regularizers = ['Y']
            activity_regularizers = ['Y']        
       
        for dataType in dataTypes:
            print('=================== ' + dataType + ' ===================')
            
            for inputType in inputTypes:
                
                print('------------------ ' + inputType + ' ------------------')
                    
                #ZmixCpv_randomequaltraintestsplit
                dataSetMethod = inputType + '_' + dataType
                
                self.modelFactory.setDataSetMethod(dataSetMethod)
                
                noOfNeurons = 53

                if inputType.find('Zmix') != -1:
                    ZmixPresent = 'Y'
                    concatenateZmix = 'Y'
                else:
                    ZmixPresent = 'N'
                    concatenateZmix = 'N'
           
                m = 3 if self.debug_mode else 6
                noOfCpvs = [item for item in range(2, m)]

                for noOfCpv in noOfCpvs:
                    for kernel_constraint in kernel_constraints:
                        for kernel_regularizer in kernel_regularizers:
                            for activity_regularizer in activity_regularizers:
                                 
                                self.executeSingleExperiment(noOfNeurons,dataSetMethod,dataType,inputType,ZmixPresent,noOfCpv,concatenateZmix,kernel_constraint,kernel_regularizer,activity_regularizer)
                       
        
    def plot_loss_physics_and_regression(self,history):
        
        f = plt.figure(figsize=(10,3))
        ax = f.add_subplot(121)
        ax2 = f.add_subplot(122)

        ax.plot(history.history['prediction_loss'], label='loss')
        ax.plot(history.history['val_prediction_loss'], label='val_loss')
        ax.set_title('Souener Prediction Loss')
        ax.set(xlabel='Epoch', ylabel='Souener Error')
        ax.legend()

        ax2.plot(history.history['physics_loss'], label='loss')
        ax2.plot(history.history['val_physics_loss'], label='val_loss')
        ax2.set_title('Physics Loss')
        ax2.set(xlabel='Epoch', ylabel='Physics Error')
        ax2.legend()    
        
