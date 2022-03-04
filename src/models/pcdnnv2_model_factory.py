# -*- coding: utf-8 -*-
"""
Created on Thu Aug	5 21:42:26 2021

@author: amol
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=45)

from tensorflow.keras.constraints import UnitNorm, Constraint
from .dnnmodel_model_factory import DNNModelFactory


class WeightsOrthogonalityConstraint(Constraint):
    def __init__(self, encoding_dim, weightage=1.0, axis=0):

        self.encoding_dim = encoding_dim
        self.weightage = weightage
        self.axis = axis

    def weights_orthogonality(self, w):
        if (self.axis == 1):
            w = tf.transpose(w)
        if (self.encoding_dim > 1):
            m = tf.matmul(tf.transpose(w), w) - tf.eye(self.encoding_dim)
            return self.weightage * tf.math.sqrt(tf.math.reduce_sum(tf.math.square(m)))
        else:
            m = tf.math.reduce_sum(w ** 2) - 1.
            return m

    def __call__(self, w):
        return self.weights_orthogonality(w)

    def get_config(self):
        return {'axis': self.axis, 'weightage': self.weightage, 'encoding_dim': self.encoding_dim}


class UncorrelatedFeaturesConstraint(Constraint):

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
        # covariance = tf.matmul(x_centered, tf.transpose(x_centered)) / tf.cast(tf.shape(x_centered)[0], tf.float32)

        return covariance

    # Constraint penalty
    def uncorrelated_feature(self, x):

        if (self.encoding_dim <= 1):
            return 0.0
        else:
            output = tf.math.reduce_sum(
                tf.math.square(self.covariance - tf.math.multiply(self.covariance, tf.eye(self.encoding_dim))))
            return output

    def __call__(self, x):
        self.covariance = self.get_covariance(x)
        return self.weightage * self.uncorrelated_feature(x)

    def get_config(self):
        return {'weightage': self.weightage, 'encoding_dim': self.encoding_dim}


# for recording in custom objects dict
def get_metric_dict():
    def log_mse(x, y): return tf.math.log(tf.math.reduce_mean((x - y) ** 2))

    def log_mae(x, y): return tf.math.log(tf.math.reduce_mean(tf.math.abs(x - y)))

    def exp_mse_mag(x, y): return tf.math.log(
        tf.math.reduce_mean((tf.math.exp(x) - tf.math.exp(y)) ** 2)) / tf.math.log(10.0)

    def exp_mae_mag(x, y): return tf.math.log(
        tf.math.reduce_mean(tf.math.abs(tf.math.exp(x) - tf.math.exp(y)))) / tf.math.log(10.0)

    def R2(yt, yp): return 1 - tf.math.reduce_mean((yp - yt) ** 2) / (tf.math.reduce_std(yt) ** 2)

    def exp_R2(yt, yp):  # these are actual names above is for convenience
        return R2(tf.math.exp(yt), tf.math.exp(yp))

    def source_true_mean(yt, yp):
        encoding_dim = yt.shape[1] // 2
        yt = yp[:, encoding_dim:]
        yp = yp[:, :encoding_dim]
        return tf.reduce_mean(yt, axis=-1)

    def source_pred_mean(yt, yp):
        encoding_dim = yt.shape[1] // 2
        yt = yp[:, encoding_dim:]
        yp = yp[:, :encoding_dim]
        return tf.reduce_mean(yp, axis=-1)

    def dynamic_source_loss(y_true, y_pred):
        assert y_true.shape[1] // 2 == y_true.shape[1] / 2
        encoding_dim = y_true.shape[1] // 2
        abs_diff = tf.math.abs(y_pred[:, :encoding_dim] - y_pred[:, encoding_dim:])
        return tf.reduce_mean(abs_diff, axis=-1)  # Note the `axis=-1`

    def R2_split(yt, yp):
        print(f'//: {yt.shape[1] // 2}, /: {yt.shape[1] / 2}')
        assert yt.shape[1] // 2 == yt.shape[1] / 2
        encoding_dim = yt.shape[1] // 2
        yt = yp[:, encoding_dim:]
        yp = yp[:, :encoding_dim]
        return 1 - tf.reduce_mean((yp - yt) ** 2, axis=-1) / (tf.math.reduce_std(yt, axis=-1) ** 2)

    return locals()


# fill globals with metric functions
globals().update(get_metric_dict())


# well tested on 2/1/22
def dynamic_source_term_pred_wrap(base_regression_model):
    from tensorflow import keras
    from tensorflow.keras import layers

    # verified to work 1/31/21
    def get_dynamic_source_truth_model(W_emb_layer):
        source_term_inputs = keras.Input(shape=(n_species,), name='source_term_input')
        source_term_truth = source_term_inputs  # keras.layers.Lambda(lambda x: keras.backend.stop_gradient(x))(source_term_inputs)
        source_term_truth = W_emb_layer(source_term_truth)
        source_term_truth_model = keras.Model(inputs=source_term_inputs, outputs=source_term_truth,
                                              name='source_term_truth')
        return source_term_truth_model

    # verified to work (when assert holds)
    # copies a models input dictionary, for reuse in a higher level model
    def copy_model_inputs(model):
        input_shape = model.input_shape
        if isinstance(model.input_shape, dict):
            # we assume coherence between input_shapes and input_names, if not true then it should be made so
            assert np.all(np.isin(model.input_names, list(model.input_shapes.keys())))
            input_shape = (model.input_shape[name] for name in model.input_names)
        elif isinstance(model.input_shape, tuple):
            input_shape = [input_shape]
        return [layers.Input(shape[1:], name=name) for name, shape in zip(model.input_names, input_shape)]

    # NOTE on indices: first squeeze is to select first input, 2nd '[1]' is to skip past batch dimension
    n_species = np.squeeze(base_regression_model.get_layer('species_input').input_shape)[1]
    encoding_dim = np.squeeze(base_regression_model.get_layer('linear_embedding').output_shape)[1]

    # copy model inputs for container model & call base_regression_model as layer/module
    copied_inputs = copy_model_inputs(base_regression_model)
    regression_outputs = base_regression_model(copied_inputs)
    dynamic_source_pred = regression_outputs['dynamic_source_prediction']

    all_species_source_inputs = keras.Input(shape=(n_species,), name='source_term_input')
    W_emb_layer = base_regression_model.get_layer('linear_embedding')
    dynamic_source_truth_model = get_dynamic_source_truth_model(W_emb_layer)
    dynamic_source_truth = dynamic_source_truth_model(all_species_source_inputs)
    dynamic_source_all = layers.Concatenate(name='dynamic_source_prediction')(
        [dynamic_source_pred, dynamic_source_truth])
    # dynamic_source_all: contains predicted and true values for computing loss
    # give it the same name as the original layer name, so it is easier to drop-in place 

    # This + source_term_truth_model facilitates dynamic source term training!
    container_model = keras.Model(
        inputs=copied_inputs + [all_species_source_inputs],
        outputs={'static_source_prediction': regression_outputs['static_source_prediction'],
                 'dynamic_source_prediction': dynamic_source_all},
        name='container_model'
    )
    return container_model


class PCDNNV2ModelFactory(DNNModelFactory):
    def __init__(self):
        super().__init__()
        self.setModelName("PCDNNV2Model")
        custom = {"PCDNNV2ModelFactory": PCDNNV2ModelFactory,
                  "UncorrelatedFeaturesConstraint": UncorrelatedFeaturesConstraint,
                  "WeightsOrthogonalityConstraint": WeightsOrthogonalityConstraint}
        custom.update(get_metric_dict())
        self.setConcreteClassCustomObject(custom)
        self.loss = 'mean_absolute_error'
        self.use_dynamic_pred = False
        return

    def get_layer_constraints(self, noOfCpv, kernel_constraint='Y', kernel_regularizer='Y', activity_regularizer='Y'):
        layer_constraints = {}
        if kernel_constraint == 'Y':
            layer_constraints['kernel_constraint'] = UnitNorm(axis=0)
        if kernel_regularizer == 'Y':
            layer_constraints['kernel_regularizer'] = WeightsOrthogonalityConstraint(noOfCpv, weightage=1., axis=0)
        if activity_regularizer == 'Y':
            layer_constraints['activity_regularizer'] = UncorrelatedFeaturesConstraint(noOfCpv, weightage=1.)
        return layer_constraints

    def addLinearLayer(self, x, noOfInputNeurons, noOfCpv, kernel_constraint='Y', kernel_regularizer='Y',
                       activity_regularizer='Y'):
        constraints = self.get_layer_constraints(noOfCpv, kernel_constraint, kernel_regularizer, activity_regularizer)
        x = layers.BatchNormalization(center=False, scale=False, name='batch_norm')(x)
        layer = layers.Dense(noOfCpv, use_bias=False, name="linear_embedding", activation="linear", **constraints)

        return layer(x)

    def build_and_compile_model(self, noOfInputNeurons, noOfOutputNeurons, noOfCpv, concatenateZmix,
                                kernel_constraint='Y', kernel_regularizer='Y', activity_regularizer='Y'):
        print(noOfInputNeurons, noOfCpv, concatenateZmix, kernel_constraint, kernel_regularizer, activity_regularizer)

        # The following 2 lines make up the Auto-encoder
        species_inputs = keras.Input(shape=(noOfInputNeurons,), name="species_input")

        # Build the regressor
        if concatenateZmix == 'Y':
            zmix = keras.Input(shape=(1,), name="zmix")
            inputs = [species_inputs, zmix]
        else:
            inputs = [species_inputs]

        x = self.addLinearModel(inputs, noOfInputNeurons, noOfCpv,
                                concatenateZmix=concatenateZmix,
                                kernel_constraint=kernel_constraint,
                                kernel_regularizer=kernel_regularizer,
                                activity_regularizer=activity_regularizer)

        source_term_pred = self.addRegressorModel(x, noOfOutputNeurons, noOfCpv)
        model = keras.Model(inputs=inputs, outputs=source_term_pred, name='emb_and_regression_model')

        if self.use_dynamic_pred:  # if use all dependents is on this will be a hybrid model
            model = dynamic_source_term_pred_wrap(model)

        opt = self.getOptimizer()

        losses = {'static_source_prediction': self.loss, 'dynamic_source_prediction': dynamic_source_loss}
        metrics = {'static_source_prediction': ['mae', 'mse', R2],
                   'dynamic_source_prediction': [R2_split, source_pred_mean, source_true_mean]}
        # for metric definitions see get_metric_dict()

        model.compile(loss=losses, optimizer=opt, metrics=metrics)

        self.model = model
        tf.keras.utils.plot_model(self.model, to_file="model.png", show_shapes=True, show_layer_names=True,
                                  rankdir="TB", expand_nested=False, dpi=96)

        return model

    # extract emb+regression model from container model (if container model is being used)
    def extractEmbRegressor(self):
        self.model = self.getEmbRegressor()
        return self.model

    def getEmbRegressor(self):
        if self.model.name == 'container_model':
            return self.model.get_layer('emb_and_regression_model')
        else:
            assert self.model.name == 'emb_and_regression_model'
            return self.model
