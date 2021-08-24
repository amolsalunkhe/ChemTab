# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 09:09:34 2021

@author: amol
"""


import time
import pandas as pd
import matplotlib.pyplot as plt
from .error_manager import ErrorManager

class PCDNNV1ExperimentExecutor:
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
        
    def executeSingleExperiment(self,noOfInputNeurons,dataSetMethod,dataType,inputType,ZmixPresent,noOfCpv,concatenateZmix):

        print("--------------------self.build_and_compile_pcdnn_v1_model----------------------")
        self.model = self.modelFactory.build_and_compile_model(noOfInputNeurons,noOfCpv,concatenateZmix)
            
        self.model.summary()

        X_train, X_test, Y_train, Y_test, rom_train, rom_test, zmix_train, zmix_test = self.dm.getTrainTestData() 
        
        ipscaler = None
        opscaler = "MinMaxScaler"
        
        #                           dataSetMethod,noOfCpvs, ipscaler, opscaler
        self.dm.createTrainTestData(dataSetMethod,noOfCpv, ipscaler, opscaler)
        
        self.modelFactory.experimentSettings = {"dataSetMethod": dataSetMethod,"ipscaler":ipscaler, "opscaler":opscaler, "noOfCpv": noOfCpv, "ZmixPresent": ZmixPresent, "concatenateZmix": concatenateZmix}
        
        X_train, X_test, Y_train, Y_test, rom_train, rom_test, zmix_train, zmix_test = self.dm.getTrainTestData() 
        
        self.fitModelAndCalcErr(X_train, Y_train, X_test, Y_test,rom_train, rom_test, zmix_train, zmix_test, self.dm.outputScaler, concatenateZmix)

        #['Model','Dataset','Cpv Type','#Cpv',"ZmixExists",'MAE','TAE','MSE','TSE','#Pts','FitTime','PredTime','MAX-MAE','MAX-TAE','MAX-MSE','MAX-TSE','MIN-MAE','MIN-TAE','MIN-MSE','MIN-TSE']

        experimentResults = [self.modelType, dataType,inputType,str(noOfCpv), ZmixPresent,str(self.df_err['MAE'].mean()),str(self.df_err['TAE'].mean()),str(self.df_err['MSE'].mean()),str(self.df_err['TSE'].mean()),str(self.df_err['#Pts'].mean()),str(self.fit_time),str(self.pred_time),str(self.df_err['MAE'].max()),str(self.df_err['TAE'].max()),str(self.df_err['MSE'].max()),str(self.df_err['TSE'].max()),str(self.df_err['MAE'].min()),str(self.df_err['TAE'].min()),str(self.df_err['MSE'].min()),str(self.df_err['TSE'].min())]

        self.df_experimentTracker.loc[len(self.df_experimentTracker)] = experimentResults        

        printStr = "self.modelType: "+ self.modelType+ " dataType: "  + dataType+ " inputType:"+inputType+ " noOfCpv:"+str(noOfCpv)+ " ZmixPresent:" + ZmixPresent + " MAE:" +str(self.df_err['MAE'].min())

        print(printStr)

        printStr = "\t"

        printStr = printStr.join(experimentResults)


    def fitModelAndCalcErr(self,X_train, Y_train, X_test, Y_test, rom_train = None, rom_test = None, zmix_train = None, zmix_test = None, Y_scaler = None, concatenateZmix = 'N'):

        fit_times = []
        
        pred_times = []
        
        errs = []
        
                
        #TODO:uncomment
        #for itr in range(1,11):
		
		#TODO:comment
        for itr in range(1,2):
            
            t = time.process_time()
            
            if concatenateZmix == 'Y':
                history = self.model.fit({"species_input":X_train, "zmix":zmix_train}, {"physics":rom_train,"prediction":Y_train},validation_split=0.2,verbose=0,epochs=100)
            else:
                history = self.model.fit({"species_input":X_train}, {"physics":rom_train,"prediction":Y_train},validation_split=0.2,verbose=0,epochs=100)
            
            #self.plot_loss_physics_and_regression(history)
            
            fit_times.append(time.process_time() - t)
        
            t = time.process_time()

            if concatenateZmix == 'Y':
                predictions = self.model.predict({"species_input":X_test, "zmix":zmix_test})
            else:
                predictions = self.model.predict({"species_input":X_test})
                
            pred_times.append(time.process_time() - t)
            
            self.predicitions = predictions
            
            Y_pred = predictions[0]
            

            if Y_scaler is not None:
                Y_pred = Y_scaler.inverse_transform(Y_pred)
                
                
            #sns.residplot(Y_pred.flatten(), getResiduals(Y_test,Y_pred))

            curr_errs = self.errManager.computeError (Y_pred, Y_test)
                
            if (len(errs) == 0) or ((len(errs) > 0) and (curr_errs[2] < self.min_mae)) :
                self.min_mae = curr_errs[2]#MAE
                self.modelFactory.saveCurrModelAsBestModel()
                    
            errs.append(curr_errs)
        
        self.fit_time = sum(fit_times)/len(fit_times)
        
        self.pred_time = sum(pred_times)/len(pred_times)
        
        #computeAndPrintError(Y_pred, Y_test)

        self.df_err = pd.DataFrame(errs, columns = ['TAE', 'TSE', 'MAE', 'MSE', 'MAPE', '#Pts'])
        
        return  

    def executeExperiments(self,dataManager, modelType, df_experimentTracker):
        self.dm = dataManager
        self.modelType = modelType
        self.df_experimentTracker = df_experimentTracker
        
        #Experiments  
        
        #TODO:uncomment
        #dataTypes = ["randomequaltraintestsplit","frameworkincludedtrainexcludedtest"]
        #inputTypes = ["AllSpeciesZmixCpv","AllSpeciesZmixPCA","AllSpeciesPurePCA","AllSpeciesSparsePCA","AllSpeciesZmixAndPurePCA","AllSpeciesZmixAndSparsePCA"]
        
        #TODO:comment
        dataTypes = ["frameworkincludedtrainexcludedtest"]
        inputTypes = ["AllSpeciesZmixAndPurePCA"]
        
        concatenateZmix = 'N'
        
        for dataType in dataTypes:
            print('=================== ' + dataType + ' ===================')
            
            for inputType in inputTypes:
                
                print('------------------ ' + inputType + ' ------------------')
                    
                #ZmixCpv_randomequaltraintestsplit
                dataSetMethod = inputType + '_' + dataType
            
                self.modelFactory.setDataSetMethod(dataSetMethod)
                
                noOfNeurons = 53

                #ZmixAnd & ZmixAll
                if inputType.find('ZmixA') != -1:
                    concatenateZmix = 'Y'
                else:
                    concatenateZmix = 'N'

                if inputType.find('Zmix') != -1:
                    ZmixPresent = 'Y'
                else:
                    ZmixPresent = 'N'
                    
                
                if inputType.find('PCA') != -1:
                    #TODO:uncomment                    
                    #noOfCpvs = [item for item in range(1, 6)]
                    
                    #TODO:comment                    
                    noOfCpvs = [item for item in range(2, 3)]
                    
                    for noOfCpv in noOfCpvs:
                        
                        self.executeSingleExperiment(noOfNeurons,dataSetMethod,dataType,inputType,ZmixPresent,noOfCpv,concatenateZmix)
                else:

                    if inputType.find('ZmixCpv') != -1:
                        noOfCpv = 1
                 
                    else:
                        noOfCpv = 53
                 
                    print('------------------ ' + str(noOfNeurons) + ' ------------------')
                    self.executeSingleExperiment(noOfNeurons,dataSetMethod,dataType,inputType,ZmixPresent,noOfCpv,concatenateZmix)
        
    
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
        