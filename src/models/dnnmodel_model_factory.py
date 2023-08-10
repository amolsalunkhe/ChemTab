# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 21:05:29 2021

@author: amol
"""
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from copy import deepcopy


class DNNModelFactory:
    def __init__(self):
        self.starter_learning_rate = 0.000001
        self.decay_steps = 100000
        self.decay_rate = 0.96
        self.clip_grad_norm = 2.5
        
        self.width = 512
        self.dropout_rate = 0.0
        self.regressor_batch_norm = False
        self.regressor_skip_connections = False
        self.activation_func='relu'
        self.model = None
        self.experimentRecord = {} 
        self.modelName = None
        self.concreteClassCustomObject = None
        self.debug_mode = False
        print("Parent DNNModelFactory Instantiated")
   
    @property # for backwards compatibility
    def experimentSettings(self):
        return self.experimentRecord

    @experimentSettings.setter # for backwards compatibility
    def experimentSettings(self, val):
        self.experimentRecord = val

    def setDataSetMethod(self,dataSetMethod):
        self.dataSetMethod = dataSetMethod

    def setModelName(self,modelName):
        self.modelName = modelName
    
    def setConcreteClassCustomObject(self, concreteClassCustomObject):
        self.concreteClassCustomObject = concreteClassCustomObject
        
    def getOptimizer(self):
        #starter_learning_rate = 0.000001

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.starter_learning_rate, decay_steps=self.decay_steps, decay_rate=self.decay_rate, staircase=True)

        # NOTE: we are testing this again for compatibility with benchmark notebook 
        opt = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=self.clip_grad_norm)
        return opt

    # reimplmeneted to non-trivial version in PCDNNv2
    # this only exists for compatibility reasons really it should only be called by addLinearLayer()
    def addLinearLayer(self,x,noOfInputNeurons, noOfCpv, **kwds):
        layer = layers.Dense(noOfCpv, use_bias=False, name="linear_embedding", activation="linear")
        return layer(x)

    def addLinearModel(self, inputs, noOfInputNeurons, noOfCpv, concatenateZmix='N', **kwds):
        """ adds PCA linear embedding 'model' (as a layer) 
            to the input tensors inside 'inputs' arg 
            (**kwds are to be passed to getLinearLaye()) """

        # make into boolean
        concatenateZmix=concatenateZmix=='Y'

        # the [1] is really important because that skips the extra (batch)
        # dimension that keras adds implicitly
        assert noOfInputNeurons == inputs[0].shape[1]
        output = self.addLinearLayer(inputs[0], noOfInputNeurons, noOfCpv, **kwds)

        linear_emb_model = keras.models.Model(inputs=inputs[0], outputs=output, name="linear_embedding")
        output = linear_emb_model(inputs[0])       
 
        # implicitly if there are 2 input layers then we want to add mix
        assert concatenateZmix == (len(inputs)>1)
        if concatenateZmix:
            zmix = inputs[1]
            #Concatenate the Linear Embedding and Zmix together
            output = layers.Concatenate(name="concatenated_zmix_linear_embedding")([zmix, output])
        return output

    def addRegressorModel(self, x, num_outputs, noOfCpv):
        """Gets layers for regression module of model (renamed from get intermediate layers)"""
        def add_regularized_dense_layer(x, layer_size):
            x = layers.Dense(layer_size, activation=self.activation_func)(x)
            if self.regressor_batch_norm: x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)
            return x

        def add_regularized_dense_module(x, layer_sizes):
            assert len(layer_sizes)==3
            skip_input = x = add_regularized_dense_layer(x, layer_sizes[0])
            x = add_regularized_dense_layer(x, layer_sizes[1])
            x = add_regularized_dense_layer(x, layer_sizes[2])
            if self.regressor_skip_connections: x = layers.Concatenate()([x, skip_input])
            return x

        # the [1:] is really important because that removes the extra (batch)
        # dimension that keras adds implicitly
        input_ = layers.Input(x.shape[1:], name='input_1')

        # # This is the simple baseline model architecture:
        #layer_sizes = [self.width//16,self.width//8,self.width//4,self.width//2]
        #layer_sizes += [self.width] + layer_sizes[::-1]

        ##layer_sizes = [16, 32, 64, 64, 32, 16]
        #output = input_
        #for size in layer_sizes:
        #    output = add_regularized_dense_layer(output, size) 
        #    #output = layers.Dense(size, activation=self.activation_func)(output)

        # TODO: refractor?
        def addRegressorOutputs(x):
            # used to be named 'prediction' (now model is named 'prediction', since it is last layer)
            static_source_pred = layers.Dense(num_outputs, name='static_source_prediction')(x)
            dynamic_source_pred = layers.Dense(noOfCpv, name='dynamic_source_prediction')(x)
            if self.batch_norm_dynamic_pred: dynamic_source_pred = layers.BatchNormalization()(dynamic_source_pred)
            return {'static_source_prediction': static_source_pred, 'dynamic_source_prediction': dynamic_source_pred}

        if self.debug_mode:
            # for debugging only
            output = add_regularized_dense_module(input_, [16,32,16])
        else:
            output = add_regularized_dense_module(input_, [self.width//16,self.width//8,self.width//4])
            output = add_regularized_dense_module(output, [self.width//2,self.width,self.width//2])
            output = add_regularized_dense_module(output, [self.width//4,self.width//8,self.width//16])

        outputs = addRegressorOutputs(output) 
        regressor_model=keras.models.Model(inputs=input_, outputs=outputs, name='regressor')

        return regressor_model(x)
      
    def saveCurrModelAsBestModel(self, path=None, experiment_results={}):
        if not path: path = './models/best_models/'+self.modelName 
        import os
        os.system('mkdir -p '+path)
        self.experimentRecord.update(experiment_results)

        # open a file, where you want to store the data
        with open(path + "/experimentRecord", "wb") as file:
            pickle.dump(self.experimentRecord, file)
       
        filePath = path +"/model" # NOTE: save_format="tf" will fix the bug related to reloading the tensorflow model & retraining!!
        tf.keras.models.save_model(self.model, filePath, overwrite=True, include_optimizer=False, save_format='tf')
        
    def openBestModel(self):
        #print("current directory" + os.getcwd())
        filePath = "./models/best_models/"+self.modelName+"/model"
        self.model = tf.keras.models.load_model(filePath, custom_objects=self.concreteClassCustomObject)
        
        # open a file, where you stored the pickled data
        file = open("./models/best_models/"+self.modelName+"/experimentRecord", "rb")
        
        self.experimentSettings = pickle.load(file)
        
        # close the file
        file.close()
        
        return self.model, self.experimentSettings 

    def getEmbRegressor(self):
        return self.model

    def getLinearEncoder(self):
        model = self.getEmbRegressor()
        model = model.get_layer('linear_embedding') # this 'layer' is actually a bonafied model
        #model.summary(expand_nested=True)
        return model
    
    def getRegressor(self):
        model = self.getEmbRegressor()
        model = model.get_layer('regressor') # this 'layer' is actually a bonafied model
        #model.summary(expand_nested=True)
        return model
