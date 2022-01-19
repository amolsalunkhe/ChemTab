# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 09:25:55 2021

@author: amol
"""
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .error_manager import ErrorManager

# returns a wrapped model factory that supports rebuilding using previous config
# it does this using dynamic inheritance, this functionality was unit tested in a notebook
def dynamic_rebuild_wrap(parent):
    class RebuildWrapper(type(parent)):
        def __init__(self, other):
            super().__init__()
            vars(self).update(vars(other))
        def build_and_compile_model(self, *args, **kwd_args):
            self._prev_config = [args, kwd_args]
            return super().build_and_compile_model(*args, **kwd_args)
        def rebuild_model(self):
            return super().build_and_compile_model(*self._prev_config[0], **self._prev_config[1])
    return RebuildWrapper(parent)

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
        self._modelFactory = None
        self.min_mae = float('inf')
        
        # override the default number of epochs used
        self.epochs_override = None
        self.n_models_override = None
        self.batch_size = 64

    @property
    def modelFactory(self):
        return self._modelFactory

    @modelFactory.setter
    def modelFactory(self, modelFactory):
        self._modelFactory = dynamic_rebuild_wrap(modelFactory)

    def setModel(self,model):
        self.model = model
    
    def setModelFactory(self,modelFactory):
        self.modelFactory = modelFactory    
    
    def getPredicitons(self):
        return self.predicitions
    
    def executeSingleExperiment(self,noOfInputNeurons,dataSetMethod,dataType,inputType,ZmixPresent,noOfCpv,concatenateZmix,kernel_constraint,kernel_regularizer,activity_regularizer,
                                ipscaler = "MinMaxScaler", opscaler = "MinMaxScaler"):
        
        print("--------------------self.build_and_compile_pcdnn_v2_model----------------------")

        self.modelFactory.debug_mode = self.debug_mode        
        self.model = self.modelFactory.build_and_compile_model(noOfInputNeurons,noOfCpv,concatenateZmix,kernel_constraint,kernel_regularizer,activity_regularizer)
            
        self.model.summary()

        self.dm.createTrainTestData(dataSetMethod,noOfCpv, ipscaler,  opscaler)

        self.modelFactory.experimentSettings = {"dataSetMethod": dataSetMethod,"ipscaler":ipscaler, "opscaler":opscaler, "noOfCpv": noOfCpv, "ZmixPresent": ZmixPresent, "concatenateZmix": concatenateZmix,"kernel_constraint":kernel_constraint,"kernel_regularizer":kernel_regularizer,"activity_regularizer":activity_regularizer, 'input_data_cols': self.dm.input_data_cols}

        X_train, X_test, Y_train, Y_test, rom_train, rom_test, zmix_train, zmix_test = self.dm.getTrainTestData() 
        
        history = self.fitModelAndCalcErr(X_train, Y_train, X_test, Y_test,None, None,
                                          zmix_train, zmix_test, self.dm.outputScaler, concatenateZmix,
                                          kernel_constraint,kernel_regularizer,activity_regularizer)
        
        #['Model','Dataset','Cpv Type','#Cpv',"ZmixExists",'MAE','TAE','MSE','TSE','#Pts','FitTime','PredTime','MAX-MAE','MAX-TAE','MAX-MSE','MAX-TSE','MIN-MAE','MIN-TAE','MIN-MSE','MIN-TSE']
        experimentResults = {'Model': self.modelType, 'Dataset':dataType, 'Cpv Type':inputType, '#Cpv':noOfCpv, 
                             'ZmixExists': ZmixPresent, '#Pts': self.df_err['#Pts'].mean(), 'FitTime': self.fit_time, 'PredTime': self.pred_time,      
                             'KernelConstraintExists': kernel_constraint, 'KernelRegularizerExists': kernel_regularizer,'ActivityRegularizerExists': activity_regularizer,
                             'OPScaler': opscaler}
        experimentResults.update(self.errManager.getExperimentErrorResults(self.df_err))
        self.df_experimentTracker = self.df_experimentTracker.append(experimentResults, ignore_index=True)

        printStr = "self.modelType: "+ self.modelType+ " dataType: "  + dataType+ " inputType:"+inputType+ " noOfCpv:"+str(noOfCpv)+ " ZmixPresent:" + ZmixPresent + " MAE:" +str(self.df_err['MAE'].min())

        print(printStr)

        printStr = "\t"

        printStr = printStr.join(experimentResults)
        return history


    def fitModelAndCalcErr(self,X_train, Y_train, X_test, Y_test, rom_train = None, rom_test = None, zmix_train = None, zmix_test = None, Y_scaler = None, concatenateZmix = 'N',kernel_constraint = 'Y',kernel_regularizer = 'Y',activity_regularizer = 'Y'):

        fit_times = []
        
        pred_times = []
        
        errs = []
      
        #if len(Y_train.shape)==1: Y_train = Y_train.reshape(-1,1)

        ## doesn't this skew the evaluation on the test data later? (e.g. outside of keras?)        
        ## TODO: remove this!
        #import warnings; warnings.warn("This is probably a bug")
        #X_train = np.concatenate((X_train, X_test),axis=0)
        #zmix_train = np.concatenate((zmix_train, zmix_test),axis=0)
        #Y_train = np.concatenate((Y_train, Y_test),axis=0)

        self.model.summary(expand_nested=True)

        Y_test_raw = Y_test # default 
        if Y_scaler is not None:
            if len(Y_test.shape)==1: Y_test = Y_test.reshape(-1,1)
            Y_test_raw = Y_scaler.inverse_transform(Y_test).flatten() 
 
        n = 3 if self.debug_mode else 6
        epochs = 5 if self.debug_mode else 100
        if self.epochs_override: epochs = self.epochs_override
        if self.n_models_override: n = self.n_models_override+1      

        for itr in range(1,n): 
            self.model = self.modelFactory.rebuild_model() 
            print(f'training model: {itr}')
            t = time.process_time()

            if concatenateZmix == 'Y':
                input_dict_train = {"species_input":X_train, "zmix":zmix_train}
                input_dict_test = {"species_input":X_test, "zmix":zmix_test}
            else:
                input_dict_train = {"species_input":X_train}
                input_dict_test = {"species_input":X_test}
            
            history = self.model.fit(input_dict_train, {"prediction":Y_train}, verbose=1,
                                     batch_size=self.batch_size, epochs=epochs, shuffle=True, 
                                     validation_data=(input_dict_test, {'prediction': Y_test}))
            #self.plot_loss_physics_and_regression(history)
            
            fit_times.append(time.process_time() - t)
        
            t = time.process_time()

            if concatenateZmix == 'Y':
                predictions = self.model.predict({"species_input":X_test, "zmix":zmix_test})
            else:
                predictions = self.model.predict({"species_input":X_test})
                
            pred_times.append(time.process_time() - t)
            
            self.predicitions = predictions
            
            Y_pred_raw = Y_pred = predictions

            if Y_scaler is not None:
                Y_pred_raw = Y_scaler.inverse_transform(Y_pred)
            #sns.residplot(Y_pred.flatten(), getResiduals(Y_test,Y_pred))

            curr_errs = self.errManager.computeError(Y_pred_raw, Y_test_raw)
                
            if curr_errs['MAE'] < self.min_mae:
                self.min_mae = curr_errs['MAE']
                self.modelFactory.saveCurrModelAsBestModel()
                self.dm.include_PCDNNV2_PCA_data(self.modelFactory, concatenateZmix=concatenateZmix)
                    
            errs.append(curr_errs)
        
        self.fit_time = sum(fit_times)/len(fit_times)
        
        self.pred_time = sum(pred_times)/len(pred_times)
        
        #computeAndPrintError(Y_pred, Y_test)

        self.df_err = pd.DataFrame(errs)

        print(self.df_err.describe())
        return history 

    def executeExperiments(self,dataManager, modelType, df_experimentTracker):
        self.dm = dataManager
        self.modelType = modelType
        self.df_experimentTracker = df_experimentTracker
        
        # #Experiments  
        #if self.debug_mode:
        #    dataTypes = ["randomequalflamesplit"]
        #    inputTypes = ["AllSpeciesAndZmix"]    
        #    opscalers = ['PositiveLogNormal', 'MinMaxScaler']
        #else:
        dataTypes = ["frameworkincludedtrainexcludedtest", "randomequalflamesplit", "randomequaltraintestsplit"]
        inputTypes = ["AllSpecies","AllSpeciesAndZmix"]
        opscalers = ['MinMaxScaler', 'QuantileTransformer', 'PositiveLogNormal', None]
        
        #concatenateZmix = 'N'
        
        kernel_constraints = ['Y','N']
        kernel_regularizers = ['Y','N']
        activity_regularizers = ['Y','N']        
       
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
                                for opscaler in opscalers:
                                    self.executeSingleExperiment(noOfNeurons,dataSetMethod,dataType,inputType,ZmixPresent,noOfCpv,concatenateZmix,kernel_constraint,
                                                                 kernel_regularizer,activity_regularizer,opscaler=opscaler)
                       
        
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
        
