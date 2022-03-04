# main.py imports
# baseline imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from models.dnnmodel_model_factory import DNNModelFactory
from models.pcdnnv2_model_factory import PCDNNV2ModelFactory
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import plot_partial_dependence
from tensorflow import keras


class NNWrapper(BaseEstimator, RegressorMixin):
    """Wraps Amol's NN classes to comform to the interface expected by SciPy"""

    def __init__(self, model, concatenate_zmix=False):
        # QUESTION: what are the things that concatenate_zmix does currently?
        super().__init__()

        assert type(concatenate_zmix) is bool

        self._input_names = [i.name for i in model.inputs]
        print(f'input names: {self._input_names}')

        # this is for finding only static_source_pred outputs
        output_names = [i.name.split('/')[0] for i in
                        model.outputs]  # the split removes the /addBias part (which I'm not sure the purpose of...)
        print(f'output names (before pruning): {output_names}')

        # here we rebulid the model with only outputs from static_source_prediction, this way the inspection classes funciton as normal 
        self._model = keras.models.Model(inputs=model.inputs,
                                         outputs=model.outputs[output_names.index('static_source_prediction')])
        self._concatenate_zmix = concatenate_zmix

        # this tells sklearn that the model is fitted apparently... (as of version 1.6.2)
        # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/validation.py
        self._I_am_fitted_ = 'nonsense_value'

    def get_XY_data(self, dm):
        """
        Extracts appropriate X & Y data from data-manager for evaluation of our model.
        (assumes that datamanager was already called to create relevant dataset)
        """
        X_train, X_test, Y_train, Y_test, zmix_train, zmix_test = dm.getTrainTestData()

        X_data = X_test
        Y_data = Y_test
        zmix = zmix_test

        # assert if zmix is in the input data that it is the first column
        lower_case_cols = [s.lower() for s in dm.input_data_cols]
        assert ('zmix' in lower_case_cols) == ('zmix' == dm.input_data_cols[0])  # == self._concatenate_zmix

        if self._concatenate_zmix:
            X_data = np.concatenate((zmix.reshape([-1, 1]), X_data), axis=1)
        extra_input_cols = ['zmix'] if self._concatenate_zmix else []

        X_data = pd.DataFrame(X_data.astype('f8'), columns=extra_input_cols + list(dm.input_data_cols))
        Y_data = pd.DataFrame(Y_data.astype('f8'), columns=dm.output_data_cols)
        return X_data, Y_data

    def predict(self, X_data):
        return self._model.predict(X_data)

    def get_params(self, deep=True):
        return self._model.get_weights()

    def fit(self, X_data, Y_data):
        self._model.fit(X_data, Y_data, batch_size=32)


# # original CustomGPR class is already compatible with scipy so we just need to add get_XY_data method...
# class GPWrapper(CustomGPR):
#     def __init__(self, gp_model):
#         # integrate gp_model into this class (i.e. this class 'becomes' gp model)
#         vars(self).update(vars(gp_model))
#
#     def get_XY_data(self, dm):
#         """
#         Extracts appropriate X & Y data from data-manager for evaluation of our model
#         (assumes that datamanager was already called to create relevant dataset)
#         """
#         X_data = pd.DataFrame(dm.X_test, columns=dm.input_data_cols)
#         Y_data = pd.DataFrame(dm.Y_test, columns=dm.output_data_cols)
#         return X_data, Y_data


class ModelInspector:
    """ Abstract model inspector (e.g. error analysis) """

    def __init__(self, model_factory, dm):
        experiment_settings = model_factory.experimentSettings
        dm.createTrainTestData(dataSetMethod=experiment_settings['dataSetMethod'],
                               numCpvComponents=experiment_settings['noOfCpv'],
                               ipscaler=experiment_settings['ipscaler'], opscaler=experiment_settings['opscaler'])

        # wrap model for use with scipy
        if isinstance(model_factory, DNNModelFactory):
            model_wrapper = NNWrapper(model_factory.getRegressor(),
                                      concatenate_zmix=experiment_settings['concatenateZmix'] == 'Y')
        # else:
        #     assert type(model_factory.model) is CustomGPR
        #     model_wrapper = GPWrapper(model_factory.model)

        self._model_factory = model_factory
        self._model = model_wrapper
        self._dm = dm
        self._X_data, self._Y_data = self._model.get_XY_data(self._dm)

        # if the model has a linear model preprocessing layer then use it to get inputs for regressor
        if isinstance(model_factory, PCDNNV2ModelFactory):
            linearAutoEncoder = model_factory.getLinearEncoder()
            X_column_names = []
            if experiment_settings['concatenateZmix'] == 'Y':
                zmix = self._X_data['zmix']
                self._X_data = self._X_data.drop('zmix', axis=1)
                self._X_data = np.concatenate((np.asarray(zmix).reshape([-1, 1]),
                                               linearAutoEncoder.predict(self._X_data)), axis=1)
                # Zmix is on the left
                X_column_names = ['zmix']
            else:
                self._X_data = linearAutoEncoder.predict(self._X_data)
            X_column_names = X_column_names + [f'cpv{i + 1}' for i in range(experiment_settings['noOfCpv'])]
            print(X_column_names)
            self._X_data = pd.DataFrame(columns=X_column_names, data=self._X_data)

    def plot_partial_dependence(self, features: list = None):
        features = list(range(min(self._X_data.shape[1], 25)))

        # this gives us the same dataframe but with only quartiles for each variable
        # thereby covering the relevant ranges but much faster
        X_chunky = self._X_data.describe().iloc[3:]
        print(X_chunky)

        plot_partial_dependence(self._model, X_chunky, features=features, target=0)

    def plot_permutation_feature_importance(self, n_repeats=5):
        def do_perm_feature_importance(model, X_data=None, Y_data=None,
                                       data_manager=None, n_repeats=30, random_state=0):
            assert not (X_data is None and Y_data is None and data_manager is None)
            if data_manager is not None:
                X_data = pd.DataFrame(data_manager.X_test, columns=data_manager.input_data_cols)
                Y_data = pd.DataFrame(data_manager.Y_test, columns=data_manager.output_data_cols)

            from sklearn.inspection import permutation_importance
            r = permutation_importance(model, X_data, Y_data,
                                       n_repeats=n_repeats,
                                       random_state=random_state,
                                       scoring='neg_mean_squared_error')

            argsort = r.importances_mean.argsort()[::-1]

            for i in argsort:
                # if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                print(f"{list(X_data.columns)[i]}; {r.importances_mean[i]:e} +/- {r.importances_std[i]:e}")
            a = list(X_data.columns[argsort])
            b = r.importances_mean[argsort]
            c = r.importances_std[argsort]
            bar = plt.bar(a, b, label='mean')
            err = plt.errorbar(a, b, yerr=c, fmt="o", color="r", label='std')
            plt.yscale('log')
            plt.title('Model\'s (Permutation) Feature Importance')
            plt.xticks(rotation=90)
            plt.legend(handles=[bar, err])
            plt.show()

        do_perm_feature_importance(self._model, self._X_data, self._Y_data, n_repeats=n_repeats)
