# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 22:13:59 2021

@author: amol
"""

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=45)

from .dnnmodel_model_factory import DNNModelFactory


class SimpleDNNModelFactory(DNNModelFactory):
    def __init__(self):
        super().__init__()
        # print("SimpleDNNModel Instantiated")
        self.setModelName("SimpleDNNModel")
        self.setConcreteClassCustomObject({"SimpleDNNModelFactory": SimpleDNNModelFactory})
        return

    def build_and_compile_model(self, noOfInputNeurons):
        inputs = keras.Input(shape=(noOfInputNeurons,), name="inputs")
        souener_pred = self.addRegressorModel(inputs)

        model = keras.Model(
            inputs=[inputs],
            outputs=[souener_pred],
        )

        opt = self.getOptimizer()

        model.compile(loss='mean_absolute_error', optimizer=opt)

        self.model = model

        return model
