import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn import gaussian_process
from sklearn.gaussian_process import GaussianProcessRegressor
import scipy.optimize
from sklearn.utils.optimize import _check_optimize_result
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
import pickle
import tensorflow as tf
from tensorflow import keras

class CustomGPR(GaussianProcessRegressor):
    def __init__(self, *args, max_iter=150,max_fun=50, gtol=1e-05, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_iter = max_iter
        self._gtol = gtol
        self._max_fun = max_fun
        
        
    #ftol = 1000000000000.0 (factr) * np.finfo(float).eps --> 0.0002220446049250313
    #,'ftol':0.0000002220446049250313
    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = scipy.optimize.minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds, options={'maxiter':self._max_iter,'maxfun':self._max_fun,'gtol': self._gtol})
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
        return theta_opt, func_min



class GPModelFactory():
	
    def __init__(self):	
        self.model = None
        
        self.experimentSettings = None
        
        self.modelName = None
        
    def getModel(self, kernel="Matern"):		
        self.modelName = "GP_" + str(kernel)
		
        if kernel == "Matern_RationalQuadratic":

			# medium term irregularities
            k1 = 0.5* Matern(length_scale=2, nu=3/2)

            k2 = 0.5* RationalQuadratic(length_scale=1.0, alpha=1.0)
            
            kernel = k1 + k2
    
        elif kernel == "RationalQuadratic":

			# medium term irregularities
            kernel = RationalQuadratic(length_scale=1.0, alpha=1.0)    

        else:
            kernel = Matern(length_scale=2, nu=3/2)

        self.model = CustomGPR(kernel=kernel)
        
        return self.model 

    def saveCurrModelAsBestModel(self):
        #print("current directory " + os.getcwd())
        
        setting_filename = "models\\best_models\\GP_experimentSettings"
        print(setting_filename)
        # open a file, where you ant to store the data
        file = open(setting_filename, "wb")
        
        # dump information to that file
        pickle.dump(self.experimentSettings, file)
        
        # close the file
        file.close()
        
        #self.model.save("models\\best_models\\"+self.modelName)
        filePath = "models\\best_models\\GP.model"
        file = open(filePath,"wb")
        pickle.dump(self.model,file)

    def openBestModel(self):
        #print("current directory" + os.getcwd())
        file = open("models\\best_models\\GP.model", "rb")
        
        self.model = pickle.load(file)
        
        # open a file, where you stored the pickled data
        file = open("models\\best_models\\GP_experimentSettings", "rb")
        
        # dump information to that file
        self.experimentSettings = pickle.load(file)
        
        # close the file
        file.close()
        
        return self.model, self.experimentSettings 
