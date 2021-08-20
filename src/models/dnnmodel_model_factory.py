# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 21:05:29 2021

@author: amol
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os
# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=45)


class DNNModelFactory:
    def __init__(self):
        self.width = 512
        self.halfwidth = 128
        self.model = None
        self.experimentSettings = None
        self.modelName = None
        self.concreteClassCustomObject = None
        print("Parent DNNModelFactory Instantiated")
    
    def setDataSetMethod(self,dataSetMethod):
        self.dataSetMethod = dataSetMethod

    def setModelName(self,modelName):
        self.modelName = modelName
    
    def setConcreteClassCustomObject(self, concreteClassCustomObject):
        self.concreteClassCustomObject = concreteClassCustomObject
        
    def getOptimizer(self):
        starter_learning_rate = 0.1
        end_learning_rate = 0.01
        decay_steps = 10000
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(starter_learning_rate, decay_steps, end_learning_rate, power=0.5)
        
        #opt = keras.optimizers.Adam(learning_rate=0.001)
        
        opt = keras.optimizers.Adam(learning_rate=learning_rate_fn)
                
        return opt
   
    def getIntermediateLayers(self,x):
        def add_regularized_dense_layer(x, layer_size, activation_func='relu', dropout_rate=0.25):
            x = layers.Dense(layer_size, activation=activation_func)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
            return x

        def add_regularized_dense_module(x, layer_sizes, activation_func='relu', dropout_rate=0.25):
            assert len(layer_sizes)==3
            skip_input = x = add_regularized_dense_layer(x, layer_sizes[0], activation_func=activation_func, dropout_rate=dropout_rate)
            x = add_regularized_dense_layer(x, layer_sizes[1], activation_func=activation_func, dropout_rate=dropout_rate)
            x = add_regularized_dense_layer(x, layer_sizes[2], activation_func=activation_func, dropout_rate=dropout_rate)
            x = layers.Concatenate()([x, skip_input])
            return x

        x = add_regularized_dense_module(x, [32,64,128])
        x = add_regularized_dense_module(x, [256,512,256])
        x = add_regularized_dense_module(x, [128,64,32])

        return x
 
    def saveCurrModelAsBestModel(self):
        #print("current directory " + os.getcwd())
        
        # open a file, where you ant to store the data
        file = open("models\\best_models\\"+self.modelName+"_experimentSettings", "wb")
        
        # dump information to that file
        pickle.dump(self.experimentSettings, file)
        
        # close the file
        file.close()
        
        #self.model.save("models\\best_models\\"+self.modelName)
        filePath = "models\\best_models\\"+self.modelName+".h5"
        tf.keras.models.save_model(self.model, filePath, overwrite=True, include_optimizer=False, save_format='h5')
        
        
    def openBestModel(self):
        #print("current directory" + os.getcwd())
        filePath = "models\\best_models\\"+self.modelName+".h5"
        self.model = tf.keras.models.load_model(filePath, custom_objects=self.concreteClassCustomObject)
        
        # open a file, where you stored the pickled data
        file = open("models\\best_models\\"+self.modelName+"_experimentSettings", "rb")
        
        # dump information to that file
        self.experimentSettings = pickle.load(file)
        
        # close the file
        file.close()
        
        return self.model, self.experimentSettings 
    
    def getLinearEncoder(self):
        
        input_layer = None
        
        linear_embedding_layer = None
        
        zmix_layer =  None
        
        for layer in self.model.layers:
            print(layer.name)            
            if layer.name == "species_input":
                input_layer = layer
            if layer.name == "zmix":
                zmix_layer = layer
            if layer.name == "linear_embedding":
                linear_embedding_layer = layer
        
        if zmix_layer is not None:
            model = tf.keras.Model ([input_layer.input],[linear_embedding_layer.output])
            
        else:
           model = tf.keras.Model ([input_layer.input],[linear_embedding_layer.output])
        
        model.summary()

        return model
    
    def getRegressor(self):
        model = None
        
        species_layer = None
        
        zmix_input = None
        
        linear_embedding_input = None
        
        concatenated_zmix_linear_embedding_layer = None
        
        for layer in self.model.layers:
            print(layer.name) 
            if layer.name == "species_input":
                species_layer = layer
            elif layer.name == "prediction":
                prediction_layer = layer
            elif layer.name == "zmix":
                zmix_input =  keras.Input(shape=(layer._keras_shape,), name="zmix_input")
            elif layer.name == "linear_embedding":
                linear_embedding_input = keras.Input(shape=(layer._keras_shape,), name="linear_embedding_input")        
            elif layer.name == "concatenated_zmix_linear_embedding":    
                x = layers.Concatenate(name="concatenated_zmix_linear_embedding")([zmix_input, linear_embedding_input])
            else:
                x = x
            
        if zmix_layer is not None:
            model = tf.keras.Model ([concatenated_zmix_linear_embedding_layer.input],[prediction_layer.output])
            
        else:
           model = tf.keras.Model ([linear_embedding_layer.input],[prediction_layer.output])
        
        '''
        self.model.layers.pop(0)
        self.model.summary()
        '''
        
        model.summary()

        return model
