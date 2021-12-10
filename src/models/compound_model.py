import os
import keras
import pickle
import pdb

def _combine_models_seq(sub_models):
    """ This is just here for completeness' sake, it combines the submodels in a 
        sequential fashion as if they were all just layers """

    model = keras.models.Sequential()
    for sub_model in sub_models:
        model.add(sub_model)
    #model.compile(optimizer='adam', loss='mse')
    return model

# TODO: figure out how to get this to store compile() info (e.g. optimizer & loss)
class CompoundKerasModel(keras.models.Model):
    def __init__(self, sub_models: list, model_combine_strategy = _combine_models_seq):
        """ pass in a list of sub-models (aka modules) & a function
            to combine them all (default is sequential combination) """
        macro_model = model_combine_strategy(sub_models)
        # kept for reference even though this class will embody the model
        
        #vars(self).update(vars(macro_model)) # model I become, much role I play --Yoda
        super().__init__(inputs=macro_model.inputs, outputs=macro_model.outputs)
        if model_combine_strategy == _combine_models_seq:
            self.compile(optimizer='adam', loss='mse')
        
        self.__sub_models = sub_models
        self.__model_combine_strategy = model_combine_strategy
        self.__macro_model = macro_model
        self.__compile_args = None  #  will be set later in compile(...)
    
    # this is a kind-of hack to make the model retain compilation settings
    def compile(self, *args, **kwds):
        self.__compile_args = (args, kwds)
        super().compile(*args, **kwds)
    
    @property
    def sub_models(self):
        return self.__sub_models
    
    @classmethod
    def load(cls, path):
        sub_models = []
        for model_fn in sorted(os.listdir(f'{path}/sub_models/')):
            sub_models.append(keras.models.load_model(f'{path}/sub_models/{model_fn}'))
        
        # since it is necessary to rebuild the model each time we pickle 
        # the actual function that does it for later use  
        with open(f'{path}/model_settings.pickle', 'rb') as f:
            settings = pickle.load(f)
        model_combine_strategy = settings['model_combine_strategy']
        compile_args = settings['compile_args']
        
        model = cls(sub_models=sub_models, model_combine_strategy=model_combine_strategy)
        if compile_args: model.compile(*compile_args[0], **compile_args[1])
        return model
        
    def save(self, path):
        os.system(f'mkdir -p {path}')
        for i, model in enumerate(self.__sub_models):
            model.save(f'{path}/sub_models/sub_model{I}')
        
        # since it is necessary to rebuild the model each time we pickle 
        # the actual function that does it for later use  
        with open(f'{path}/model_settings.pickle', 'wb') as f:
            settings = {'model_combine_strategy': self.__model_combine_strategy,
                        'compile_args': self.__compile_args}
            pickle.dump(settings, f)
