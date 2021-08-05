# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 09:25:55 2021

@author: amol
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import EarlyStopping
import time
import pandas as pd
import matplotlib.pyplot as plt
# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=45)

from tensorflow.keras import regularizers, activations, initializers, constraints, Sequential
from tensorflow.keras.constraints import UnitNorm, Constraint
from tensorflow.keras import backend as K




class WeightsOrthogonalityConstraint (Constraint):
    def __init__(self, encoding_dim, weightage = 1.0, axis = 0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage
        self.axis = axis
        
    def weights_orthogonality(self, w):
        if(self.axis==1):
            w = tf.transpose(w)
        if(self.encoding_dim > 1):
            m = tf.matmul(tf.transpose(w), w) - tf.eye(self.encoding_dim)
            return self.weightage * tf.math.sqrt(tf.math.reduce_sum(tf.math.square(m)))
        else:
            m = tf.math.reduce_sum(w ** 2) - 1.
            return m

    def __call__(self, w):
        return self.weights_orthogonality(w)
    
    def get_config(self):
        return {'encoding_dim':self.encoding_dim}
    

class UncorrelatedFeaturesConstraint (Constraint):

    def __init__(self, encoding_dim, weightage=1.0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage

    def get_covariance(self, x):
        x_centered_list = []

        for i in range(self.encoding_dim):
            x_centered_list.append(x[:, i] - tf.math.reduce_mean(x[:, i]))

        x_centered = tf.stack(x_centered_list)
        covariance = tf.matmul(x_centered, tf.transpose(x_centered)) / \
            tf.cast(x_centered.get_shape()[0], tf.float32)

        return covariance

    # Constraint penalty
    def uncorrelated_feature(self, x):
        if(self.encoding_dim <= 1):
            return 0.0
        else:
            output = tf.math.reduce_sum(tf.math.square(
                self.covariance - tf.math.multiply(self.covariance, tf.eye(self.encoding_dim))))
            return output

    def __call__(self, x):
        self.covariance = self.get_covariance(x)
        return self.weightage * self.uncorrelated_feature(x)
    
    def get_config(self):
        return {'weightage': self.weightage, 'encoding_dim':self.encoding_dim}


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
        self.width = 512
        self.halfwidth = 128 
        self.predicitions = None
    
    def setModel(self,model):
        self.model = model
        
    def getPredicitons(self):
        return self.predicitions

    def build_and_compile_pcdnn_v1_wo_zmix_model(self,noOfInputNeurons,noOfCpv,kernel_constraint='Y',kernel_regularizer='Y',activity_regularizer='Y'):

        print (noOfInputNeurons,noOfCpv,kernel_constraint,kernel_regularizer,activity_regularizer)
        
        species_inputs = keras.Input(shape=(noOfInputNeurons,), name="species_input")
        
        x = self.getLinearLayer(noOfInputNeurons,noOfCpv,kernel_constraint,kernel_regularizer,activity_regularizer)(species_inputs)
        
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(32, activation="relu")(x)
        #Predict the source energy
        souener_pred = layers.Dense(1, name="prediction")(x)

        model = keras.Model(inputs=[species_inputs],outputs=[souener_pred],)

        opt = keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(loss='mean_absolute_error',optimizer=opt)
        
        return model

    
    def getLinearLayer(self,noOfInputNeurons,noOfCpv,kernel_constraint='Y',kernel_regularizer='Y',activity_regularizer='Y'):
        
        if kernel_constraint=='Y'and kernel_regularizer =='N' and activity_regularizer =='N':
            layer = layers.Dense(noOfCpv, activation="linear",kernel_constraint=UnitNorm(axis=0))
        
        elif kernel_constraint=='N'and kernel_regularizer =='Y' and activity_regularizer =='N':
            layer = layers.Dense(noOfCpv, activation="linear",kernel_regularizer=WeightsOrthogonalityConstraint(noOfCpv, weightage=1., axis=0))
            
        elif kernel_constraint=='N'and kernel_regularizer =='N' and activity_regularizer =='Y':
            layer = layers.Dense(noOfCpv, activation="linear",activity_regularizer=UncorrelatedFeaturesConstraint(noOfCpv, weightage=1.))
        
        elif kernel_constraint=='Y'and kernel_regularizer =='Y' and activity_regularizer =='N':
            layer = layers.Dense(noOfCpv, activation="linear",kernel_constraint=UnitNorm(axis=0),kernel_regularizer=WeightsOrthogonalityConstraint(noOfCpv, weightage=1., axis=0))
        
        elif kernel_constraint=='Y'and kernel_regularizer =='N' and activity_regularizer =='Y':
            layer = layers.Dense(noOfCpv, activation="linear",kernel_constraint=UnitNorm(axis=0),activity_regularizer=UncorrelatedFeaturesConstraint(noOfCpv, weightage=1.))
        
        elif kernel_constraint=='N'and kernel_regularizer =='Y' and activity_regularizer =='Y':
            layer = layers.Dense(noOfCpv, activation="linear",kernel_regularizer=WeightsOrthogonalityConstraint(noOfCpv, weightage=1., axis=0),activity_regularizer=UncorrelatedFeaturesConstraint(noOfCpv, weightage=1.))
        
        elif kernel_constraint=='N'and kernel_regularizer =='N' and activity_regularizer =='N':
            layer = layers.Dense(noOfCpv, activation="linear")
                
        else:
            layer = layers.Dense(noOfCpv, activation="linear",kernel_constraint=UnitNorm(axis=0),kernel_regularizer=WeightsOrthogonalityConstraint(noOfCpv, weightage=1., axis=0),activity_regularizer=UncorrelatedFeaturesConstraint(noOfCpv, weightage=1.))
        return layer

    def build_and_compile_pcdnn_v1_with_zmix_model(self,noOfInputNeurons,noOfCpv,kernel_constraint='Y',kernel_regularizer='Y',activity_regularizer='Y'):

        species_inputs = keras.Input(shape=(noOfInputNeurons,), name="species_input")

        Zmix = keras.Input(shape=(1,), name="Zmix")

        x = self.getLinearLayer(noOfInputNeurons,noOfCpv,kernel_constraint,kernel_regularizer,activity_regularizer)(species_inputs)

        #Concatenate the Linear Embedding and Zmix together
        x = layers.Concatenate()([Zmix, x])

        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(32, activation="relu")(x)
        #Predict the source energy
        souener_pred = layers.Dense(1, name="prediction")(x)

        model = keras.Model(inputs=[species_inputs,Zmix],outputs=[souener_pred],)

        opt = keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(loss='mean_absolute_error',optimizer=opt)
        
        return model
    
    
    def executeSingleExperiment(self,noOfInputNeurons,dataSetMethod,dataType,inputType,ZmixPresent,noOfCpv,concatenateZmix,kernel_constraint,kernel_regularizer,activity_regularizer):
        
        
        #                           dataSetMethod,noOfCpvs, ipscaler, opscaler
        self.dm.createTrainTestData(dataSetMethod,noOfCpv, "MinMaxScaler", "MinMaxScaler")
        
        if concatenateZmix == 'N':
            print("--------------------self.build_and_compile_pcdnn_v1_wo_zmix_model----------------------")
            self.model = self.build_and_compile_pcdnn_v1_wo_zmix_model(noOfInputNeurons,noOfCpv,kernel_constraint,kernel_regularizer,activity_regularizer)
            
        else:
            self.model = self.build_and_compile_pcdnn_v1_with_zmix_model(noOfInputNeurons,noOfCpv,kernel_constraint,kernel_regularizer,activity_regularizer)

        
        self.model.summary()

        #(self,X_train, Y_train, X_test, Y_test, rom_train = None, rom_test = None, zmix_train = None, zmix_test = None, Y_scaler = None)
        if self.dm.inputScaler is not None:
            X_train = self.dm.X_scaled_train
            X_test = self.dm.X_scaled_test
            zmix_train = self.dm.zmix_scaled_train
            zmix_test = self.dm.zmix_scaled_test
        else:
            X_train = self.dm.X_train
            X_test = self.dm.X_test
            zmix_train = self.dm.zmix_train
            zmix_test = self.dm.zmix_test
        
        if self.dm.outputScaler is not None:
            Y_train = self.dm.Y_scaled_train
            Y_test = self.dm.Y_scaled_test
        else:
            Y_train = self.dm.Y_train
            Y_test = self.dm.Y_test

            
        self.fitModelAndCalcErr(X_train, Y_train, X_test, Y_test,None, None, zmix_train, zmix_test, self.dm.outputScaler, concatenateZmix,kernel_constraint,kernel_regularizer,activity_regularizer)

        #['Model','Dataset','Cpv Type','#Cpv',"ZmixExists",'MAE','TAE','MSE','TSE','#Pts','FitTime','PredTime','MAX-MAE','MAX-TAE','MAX-MSE','MAX-TSE','MIN-MAE','MIN-TAE','MIN-MSE','MIN-TSE']

        experimentResults = [self.modelType, dataType,inputType,str(noOfCpv), ZmixPresent,kernel_constraint,kernel_regularizer,activity_regularizer,str(self.df_err['MAE'].mean()),str(self.df_err['TAE'].mean()),str(self.df_err['MSE'].mean()),str(self.df_err['TSE'].mean()),str(self.df_err['#Pts'].mean()),str(self.fit_time),str(self.pred_time),str(self.df_err['MAE'].max()),str(self.df_err['TAE'].max()),str(self.df_err['MSE'].max()),str(self.df_err['TSE'].max()),str(self.df_err['MAE'].min()),str(self.df_err['TAE'].min()),str(self.df_err['MSE'].min()),str(self.df_err['TSE'].min())]

        self.df_experimentTracker.loc[len(self.df_experimentTracker)] = experimentResults        

        printStr = "self.modelType: "+ self.modelType+ " dataType: "  + dataType+ " inputType:"+inputType+ " noOfCpv:"+str(noOfCpv)+ " ZmixPresent:" + ZmixPresent + " MAE:" +str(self.df_err['MAE'].min())

        print(printStr)

        printStr = "\t"

        printStr = printStr.join(experimentResults)



    def fitModelAndCalcErr(self,X_train, Y_train, X_test, Y_test, rom_train = None, rom_test = None, zmix_train = None, zmix_test = None, Y_scaler = None, concatenateZmix = 'N',kernel_constraint = 'Y',kernel_regularizer = 'Y',activity_regularizer = 'Y'):

        fit_times = []
        
        pred_times = []
        
        errs = []
        
        #TODO:uncomment    
        #for itr in range(1,11):
        
        #TODO:comment    
        for itr in range(1,2):    
            
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

            errs.append(self.computeError (Y_pred, Y_test))
        
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
        #inputTypes = ["AllSpecies","AllSpeciesAndZmix"]
        

        #TODO:comment
        dataTypes = ["randomequaltraintestsplit"]
        inputTypes = ["AllSpeciesAndZmix"]
        
        
        
        concatenateZmix = 'N'
        
        #TODO:uncomment
        #kernel_constraints = ['Y','N']
        #kernel_regularizers = ['Y','N']
        #activity_regularizers = ['Y','N']        
       
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
                
                noOfNeurons = 53

                if inputType.find('Zmix') != -1:
                    ZmixPresent = 'Y'
                    concatenateZmix = 'Y'
                else:
                    ZmixPresent = 'N'
                    concatenateZmix = 'N'
                
                #TODO:uncomment    
                #noOfCpvs = [item for item in range(2, 6)]

                #TODO:comment    
                noOfCpvs = [item for item in range(2, 3)]

                for noOfCpv in noOfCpvs:
                    for kernel_constraint in kernel_constraints:
                        for kernel_regularizer in kernel_regularizers:
                            for activity_regularizer in activity_regularizers:
                                 
                                self.executeSingleExperiment(noOfNeurons,dataSetMethod,dataType,inputType,ZmixPresent,noOfCpv,concatenateZmix,kernel_constraint,kernel_regularizer,activity_regularizer)
                       
        
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
        