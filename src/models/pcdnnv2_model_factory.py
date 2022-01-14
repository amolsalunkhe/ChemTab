# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 21:42:26 2021

@author: amol
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import EarlyStopping
# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=45)

from tensorflow.keras import regularizers, activations, initializers, constraints, Sequential
from tensorflow.keras.constraints import UnitNorm, Constraint
from tensorflow.keras import backend as K
from .dnnmodel_model_factory import DNNModelFactory

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
        return {'axis':self.axis,'weightage':self.weightage,'encoding_dim':self.encoding_dim}
    

class UncorrelatedFeaturesConstraint (Constraint):

    def __init__(self, encoding_dim, weightage=1.0):
        self.encoding_dim = encoding_dim
        
        self.weightage = weightage

        self.covariance = None
        
    def get_covariance(self, x):
        x_centered_list = []

        for i in range(self.encoding_dim):
            x_centered_list.append(x[:, i] - tf.math.reduce_mean(x[:, i]))

        x_centered = tf.stack(x_centered_list)
        
        covariance = tf.matmul(x_centered, tf.transpose(x_centered)) / tf.cast(x_centered.get_shape()[0], tf.float32)
        #covariance = tf.matmul(x_centered, tf.transpose(x_centered)) / tf.cast(tf.shape(x_centered)[0], tf.float32)

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

class PCDNNV2ModelFactory(DNNModelFactory):
    def __init__(self):
        super().__init__()
        self.setModelName("PCDNNV2Model")
        self.setConcreteClassCustomObject({"PCDNNV2ModelFactory": PCDNNV2ModelFactory,"UncorrelatedFeaturesConstraint":UncorrelatedFeaturesConstraint,"WeightsOrthogonalityConstraint":WeightsOrthogonalityConstraint}) 
        self.loss = 'mean_absolute_error'
        return

    def getLinearLayer(self,noOfInputNeurons,noOfCpv,kernel_constraint='Y',kernel_regularizer='Y',activity_regularizer='Y'):

        if kernel_constraint=='Y'and kernel_regularizer =='N' and activity_regularizer =='N':
            layer = layers.Dense(noOfCpv, name="linear_embedding", activation="linear",kernel_constraint=UnitNorm(axis=0))
        
        elif kernel_constraint=='N'and kernel_regularizer =='Y' and activity_regularizer =='N':
            layer = layers.Dense(noOfCpv, name="linear_embedding", activation="linear",kernel_regularizer=WeightsOrthogonalityConstraint(noOfCpv, weightage=1., axis=0))
            
        elif kernel_constraint=='N'and kernel_regularizer =='N' and activity_regularizer =='Y':
            layer = layers.Dense(noOfCpv, name="linear_embedding", activation="linear",activity_regularizer=UncorrelatedFeaturesConstraint(noOfCpv, weightage=1.))
        
        elif kernel_constraint=='Y'and kernel_regularizer =='Y' and activity_regularizer =='N':
            layer = layers.Dense(noOfCpv, name="linear_embedding", activation="linear",kernel_constraint=UnitNorm(axis=0),kernel_regularizer=WeightsOrthogonalityConstraint(noOfCpv, weightage=1., axis=0))
        
        elif kernel_constraint=='Y'and kernel_regularizer =='N' and activity_regularizer =='Y':
            layer = layers.Dense(noOfCpv, name="linear_embedding", activation="linear",kernel_constraint=UnitNorm(axis=0),activity_regularizer=UncorrelatedFeaturesConstraint(noOfCpv, weightage=1.))
        
        elif kernel_constraint=='N'and kernel_regularizer =='Y' and activity_regularizer =='Y':
            layer = layers.Dense(noOfCpv, name="linear_embedding", activation="linear",kernel_regularizer=WeightsOrthogonalityConstraint(noOfCpv, weightage=1., axis=0),activity_regularizer=UncorrelatedFeaturesConstraint(noOfCpv, weightage=1.))
        
        elif kernel_constraint=='N'and kernel_regularizer =='N' and activity_regularizer =='N':
            layer = layers.Dense(noOfCpv, name="linear_embedding", activation="linear")
                
        else:
            layer = layers.Dense(noOfCpv, name="linear_embedding", activation="linear",kernel_constraint=UnitNorm(axis=0),kernel_regularizer=WeightsOrthogonalityConstraint(noOfCpv, weightage=1., axis=0),activity_regularizer=UncorrelatedFeaturesConstraint(noOfCpv, weightage=1.))
        return layer

 
    def build_and_compile_model(self,noOfInputNeurons,noOfOutputNeurons,noOfCpv,concatenateZmix,kernel_constraint='Y',kernel_regularizer='Y',activity_regularizer='Y'):
        print(noOfInputNeurons,noOfCpv,concatenateZmix,kernel_constraint,kernel_regularizer,activity_regularizer)
        
        #The following 2 lines make up the Auto-encoder
        species_inputs = keras.Input(shape=(noOfInputNeurons,), name="species_input")
        
        #Build the regressor
        if concatenateZmix == 'Y':
            zmix = keras.Input(shape=(1,), name="zmix")
            inputs = [species_inputs,zmix]
        else:
            inputs = [species_inputs]
       
        x = self.addLinearModel(inputs, noOfInputNeurons, noOfCpv,
                                concatenateZmix=concatenateZmix,
                                kernel_constraint=kernel_constraint,
                                kernel_regularizer=kernel_regularizer,
                                activity_regularizer=activity_regularizer)
 
        souener_pred = self.addRegressorModel(x, noOfOutputNeurons)
        model = keras.Model(inputs=inputs,outputs=souener_pred)

        opt = self.getOptimizer()
        
        def log_mse(x,y): return tf.math.log(tf.math.reduce_mean((x-y)**2))
        def log_mae(x,y): return tf.math.log(tf.math.reduce_mean(tf.math.abs(x-y)))
        def exp_mse_mag(x,y): return tf.math.log(tf.math.reduce_mean((tf.math.exp(x)-tf.math.exp(y))**2))/tf.math.log(10.0)
        def exp_mae_mag(x,y): return tf.math.log(tf.math.reduce_mean(tf.math.abs(tf.math.exp(x)-tf.math.exp(y))))/tf.math.log(10.0)
        def R2(yt,yp): return 1-tf.math.reduce_mean((yp-yt)**2)/(tf.math.reduce_std(yt)**2)
        def exp_R2(yt,yp): # these are actual names above is for convenience
            return R2(tf.math.exp(yt), tf.math.exp(yp))
        
        model.compile(loss=self.loss,optimizer=opt, metrics=['mae', 'mse',  exp_mse_mag, exp_mae_mag, exp_R2, R2])
        
        self.model = model
        tf.keras.utils.plot_model(self.model,to_file="model.png",show_shapes=True,show_layer_names=True,rankdir="TB",expand_nested=False,dpi=96)
    
        return model

