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
        self.dropout_rate = 0.5
        self.activation_func='relu'
        #self.halfwidth = 128
        self.model = None
        self.experimentSettings = None
        self.modelName = None
        self.concreteClassCustomObject = None
        self.debug_mode = False
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
        
        opt = keras.optimizers.Adam(learning_rate=learning_rate_fn, clipnorm=5)
                
        return opt

    # reimplmeneted to non-trivial version in PCDNNv2
    # this only exists for compatibility reasons really it should only be called by addLinearLayer()
    def getLinearLayer(self,noOfInputNeurons, noOfCpv, **kwds):
        layer = layers.Dense(noOfCpv, name="linear_embedding", activation="linear")
        return layer

    def addLinearModel(self, inputs, noOfInputNeurons, noOfCpv, concatenateZmix='N', **kwds):
        """ adds PCA linear embedding 'model' (as a layer) 
            to the input tensors inside 'inputs' arg 
            (**kwds are to be passed to getLinearLaye()) """

        # make into boolean
        concatenateZmix=concatenateZmix=='Y'

        # the [1] is really important because that skips the extra (batch)
        # dimension that keras adds implicitly
        assert noOfInputNeurons == inputs[0].shape[1]
        output = self.getLinearLayer(noOfInputNeurons, noOfCpv, **kwds)(inputs[0])

        linear_emb_model = keras.models.Model(inputs=inputs[0], outputs=output, name="linear_embedding")
        output = linear_emb_model(inputs[0])       
 
        # implicitly if there are 2 input layers then we want to add mix
        assert concatenateZmix == (len(inputs)>1)
        if concatenateZmix:
            zmix = inputs[1]
 
            #Concatenate the Linear Embedding and Zmix together
            output = layers.Concatenate(name="concatenated_zmix_linear_embedding")([zmix, output])

        #linear_emb_model = keras.models.Model(inputs=inputs, outputs=output, name="linear_embedding")
        return output

    def addRegressorModel(self, x):
        """Gets layers for regression module of model (renamed from get intermediate layers)"""
        def add_regularized_dense_layer(x, layer_size):
            x = layers.Dense(layer_size, activation=self.activation_func)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)
            return x

        def add_regularized_dense_module(x, layer_sizes):
            assert len(layer_sizes)==3
            skip_input = x = add_regularized_dense_layer(x, layer_sizes[0])
            x = add_regularized_dense_layer(x, layer_sizes[1])
            x = add_regularized_dense_layer(x, layer_sizes[2])
            x = layers.Concatenate()([x, skip_input])
            return x

        # the [1:] is really important because that removes the extra (batch)
        # dimension that keras adds implicitly
        input_ = layers.Input(x.shape[1:], name='input_1')

        # Amol's original arch, doesn't work because we need x to be preserved
        #x = layers.Dense(32, activation="relu")(input_)
        #x = layers.Dense(64, activation="relu")(x)
        #x = layers.Dense(128, activation="relu")(x)
        #x = layers.Dense(256, activation="relu")(x)
        #x = layers.Dense(512, activation="relu")(x)
        #x = layers.Dense(256, activation="relu")(x)
        #x = layers.Dense(128, activation="relu")(x)
        #x = layers.Dense(64, activation="relu")(x)
        #x = layers.Dense(32, activation="relu")(x)
        #output = x

        layer_sizes = [32,64,128,256,512,256,128,64,32]
        #layer_sizes = [16, 32, 64, 64, 32, 16]
        output = input_
        for size in layer_sizes:
            output = layers.Dense(size, activation='relu')(output)
        
        #if self.debug_mode: 
        #    # for debugging only
        #    output = add_regularized_dense_module(input_, [16,32,16])
        #else:
        #    output = add_regularized_dense_module(input_, [self.width//16,self.width//8,self.width//4])
        #    output = add_regularized_dense_module(output, [self.width//2,self.width,self.width//2])
        #    output = add_regularized_dense_module(output, [self.width//4,self.width//8,self.width//16])

        # used to be named 'prediction' (now model is named 'prediction', since it is last layer)
        souener_pred = layers.Dense(1)(output)
        regressor_model=keras.models.Model(inputs=input_, outputs=souener_pred, name='prediction')

        return regressor_model(x)
      
    def saveCurrModelAsBestModel(self):
        #print("current directory " + os.getcwd())

        import os
        os.system('mkdir -p ./models/best_models/'+self.modelName)

        # open a file, where you ant to store the data
        file = open("./models/best_models/"+self.modelName+"/experimentSettings", "wb")
        
        # dump information to that file
        pickle.dump(self.experimentSettings, file)
        
        # close the file
        file.close()
        
        #self.model.save("models\\best_models\\"+self.modelName)
        filePath = "./models/best_models/"+self.modelName+"/model.h5"
        tf.keras.models.save_model(self.model, filePath, overwrite=True, include_optimizer=False, save_format='h5')
        
        
    def openBestModel(self):
        #print("current directory" + os.getcwd())
        filePath = "./models/best_models/"+self.modelName+"/model.h5"
        self.model = tf.keras.models.load_model(filePath, custom_objects=self.concreteClassCustomObject)
        
        # open a file, where you stored the pickled data
        file = open("./models/best_models/"+self.modelName+"/experimentSettings", "rb")
        
        # dump information to that file
        self.experimentSettings = pickle.load(file)
        
        # close the file
        file.close()
        
        return self.model, self.experimentSettings 
    
    def getLinearEncoder(self):
        model_layers = {layer.name: layer for layer in self.model.layers}
        model = model_layers['linear_embedding'] # this 'layer' is actually a bonafied model
        model.summary()

        return model
    
    def getRegressor(self):
        model_layers = {layer.name: layer for layer in self.model.layers}
        model = model_layers['prediction'] # unfortunately regressor is named this for compatibility with older code
        model.summary()

        return model
