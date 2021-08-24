# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 21:57:24 2021

@author: amol
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=45)
from .dnnmodel_model_factory import DNNModelFactory

class PCDNNV1ModelFactory(DNNModelFactory):
    def __init__(self):
        self.setModelName("PCDNNV1Model")
        self.setConcreteClassCustomObject({"PCDNNV1ModelFactory": PCDNNV1ModelFactory})
        return

    def build_and_compile_model(self,noOfInputNeurons,noOfCpv,concatenateZmix):

        species_inputs = keras.Input(shape=(noOfInputNeurons,), name="species_input")

        if concatenateZmix == 'Y':
            zmix = keras.Input(shape=(1,), name="zmix")
            inputs = [species_inputs,zmix]
        else:
            inputs = [species_inputs]

        linear_reduced_dims = self.addLinearModel(inputs, noOfInputNeurons, noOfCpv,
                                concatenateZmix=concatenateZmix)
 
        souener_pred = self.addRegressorModel(linear_reduced_dims)

        physics_pred = layers.Dense(noOfCpv, name="physics")(linear_reduced_dims)

        model = keras.Model(inputs=inputs, outputs=[souener_pred, physics_pred])

        opt = self.getOptimizer()

        model.compile(loss={"physics": keras.losses.MeanAbsoluteError(),"prediction": keras.losses.MeanAbsoluteError()},loss_weights=[2.0, 0.2],optimizer=opt)

        self.model = model

        return model
