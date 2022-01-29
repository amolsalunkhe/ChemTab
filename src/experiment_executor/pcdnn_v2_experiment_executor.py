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
        self.predictions = None
        self.errManager = ErrorManager()
        self._modelFactory = None
        self.min_mae = float('inf')
        
        # override the default number of epochs used
        self.epochs_override = None
        self.n_models_override = None
        self.batch_size = 64
        self.use_dependants = False

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
    
    def getPredictons(self):
        return self.predictions

    
    def executeSingleExperiment(self,noOfInputNeurons,dataSetMethod,dataType,inputType,ZmixPresent,noOfCpv,concatenateZmix,kernel_constraint,kernel_regularizer,activity_regularizer, ipscaler = "MinMaxScaler", opscaler = "MinMaxScaler"):
        
        print("--------------------self.build_and_compile_pcdnn_v2_model----------------------")

        method_parts = dataSetMethod.split('_')
        dependants = method_parts[2]

        self.dm.createTrainTestData(dataSetMethod, noOfCpv, ipscaler,  opscaler)
        #self.errManager.set_souener_index(self.dm) # confirm/set index of souener in output
        X_train, X_test, Y_train, Y_test, rom_train, rom_test, zmix_train, zmix_test = self.dm.getTrainTestData()

        noOfOutputNeurons = Y_test.shape[1]
        
        self.modelFactory.debug_mode = self.debug_mode        
        self.model = self.modelFactory.build_and_compile_model(noOfInputNeurons,noOfOutputNeurons,noOfCpv,concatenateZmix,
                                                               kernel_constraint,kernel_regularizer,activity_regularizer)
            
        self.model.summary()

        self.modelFactory.experimentSettings = {"dataSetMethod": dataSetMethod,"ipscaler":ipscaler, "opscaler":opscaler, "noOfCpv": noOfCpv, "ZmixPresent": ZmixPresent, "concatenateZmix": concatenateZmix,"kernel_constraint":kernel_constraint,"kernel_regularizer":kernel_regularizer,"activity_regularizer":activity_regularizer, 'input_data_cols': self.dm.input_data_cols}
        
        history = self.fitModelAndCalcErr(X_train, Y_train, X_test, Y_test,None, None,
                                          zmix_train, zmix_test, self.dm.outputScaler, concatenateZmix,
                                          kernel_constraint,kernel_regularizer,activity_regularizer)
        
        #['Model','Dataset','Cpv Type','#Cpv',"ZmixExists",'MAE','TAE','MSE','TSE','#Pts','FitTime','PredTime','MAX-MAE','MAX-TAE','MAX-MSE','MAX-TSE','MIN-MAE','MIN-TAE','MIN-MSE','MIN-TSE']
        experimentResults = {'Model': self.modelType, 'Dataset':dataType, 'Cpv Type':inputType, 'Dependants': dependants,  '#Cpv':noOfCpv, 
                             'ZmixExists': ZmixPresent, '#Pts': self.df_err['#Pts'].mean(), 'FitTime': self.fit_time, 'PredTime': self.pred_time,      
                             'KernelConstraintExists': kernel_constraint, 'KernelRegularizerExists': kernel_regularizer,'ActivityRegularizerExists': activity_regularizer,
                             'OPScaler': opscaler}
        experimentResults.update(self.errManager.getExperimentErrorResults(self.df_err))
        self.df_experimentTracker = self.df_experimentTracker.append(experimentResults, ignore_index=True)

        printStr = "self.modelType: "+ self.modelType+ " dataType: "  + dataType+ " inputType:"+inputType+ " noOfCpv:"+str(noOfCpv)+ " ZmixPresent:" + ZmixPresent + " MAE:" +str(self.df_err['MAE'].min())

        print(printStr)

        printStr = "\t"

        printStr = printStr.join(experimentResults)
        return self.df_err


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

        assert len(Y_test.shape) == len(Y_train.shape)
        if len(Y_test.shape)==1: 
            Y_test = Y_test.reshape(-1,1)
            Y_train = Y_train.reshape(-1,1)
        Y_test_raw = Y_test # default 
        if Y_scaler is not None:
            Y_test_raw = Y_scaler.inverse_transform(Y_test) 
 
        n = 2 if self.debug_mode else 3
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

            # [1] skips batch dimension
            dynamic_size = self.model.output_shape['dynamic_source_prediction'][1]
            dummy_source_term_data = np.zeros(shape=(Y_train.shape[0],dynamic_size))            

            history = self.model.fit(input_dict_train, {"static_source_prediction":Y_train, 'dynamic_source_prediction': dummy_source_term_data}, verbose=1,
                                     batch_size=self.batch_size, epochs=epochs, shuffle=True, 
                                     validation_data=(input_dict_test, {'static_source_prediction': Y_test}))
            #self.plot_loss_physics_and_regression(history)
            
            fit_times.append(time.process_time() - t)
        
            t = time.process_time()

            if concatenateZmix == 'Y':
                predictions = self.model.predict({"species_input":X_test, "zmix":zmix_test})
            else:
                predictions = self.model.predict({"species_input":X_test})
                
            pred_times.append(time.process_time() - t)
            
            self.predictions = predictions
            
            # select only static source prediction output, to match scaler
            Y_pred_raw = Y_pred = predictions['static_source_prediction']

            if Y_scaler is not None:
                Y_pred_raw = Y_scaler.inverse_transform(Y_pred)
            #sns.residplot(Y_pred.flatten(), getResiduals(Y_test,Y_pred))

            # select only static source-ener term for official error computation
            curr_errs = self.errManager.computeError(Y_pred_raw[:, self.dm.souener_index], Y_test_raw[:, self.dm.souener_index])
                
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
        
        dependents = 'AllDependants' if self.use_dependants else 'NoDependants'
         #['AllDependants', 'NoDependants'] #"souener","souspecO2", "souspecCO", "souspecCO2", "souspecH2O", "souspecOH", "souspecH2", "souspecCH4"]

        # #Experiments  
        #if self.debug_mode:
        #    dataTypes = ["randomequalflamesplit"]
        #    inputTypes = ["AllSpeciesAndZmix"]    
        #    opscalers = ['PositiveLogNormal', 'MinMaxScaler']
        #else:
        dataTypes = ["randomequalflamesplit", "randomequaltraintestsplit"]#, "frameworkincludedtrainexcludedtest"]
        inputTypes = ["AllSpecies","AllSpeciesAndZmix"]
        opscalers = ['MinMaxScaler', 'PositiveLogNormal']#, 'QuantileTransformer', None]
        
        #concatenateZmix = 'N'
        
        kernel_constraints = ['Y','N']
        kernel_regularizers = ['Y','N']
        activity_regularizers = ['Y','N']        
       
        for dataType in dataTypes:
            print('=================== ' + dataType + ' ===================')
            
            for inputType in inputTypes:
                
                print('------------------ ' + inputType + ' ------------------')
                    
                #ZmixCpv_randomequaltraintestsplit
                dataSetMethod = inputType + '_' + dataType + '_' + dependents
                
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
        
