import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn import gaussian_process
from sklearn.gaussian_process import GaussianProcessRegressor
import scipy.optimize
from sklearn.utils.optimize import _check_optimize_result
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, RationalQuadratic, ExpSineSquared

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



class GPModel():
	def __init__(self):	
		self.model = None
		
	def getModel(self, kernel="Matern"):		
		if kernel == "Matern_RationalQuadratic":

			# medium term irregularities
			k1 = 0.5* Matern(length_scale=2, nu=3/2)
			k2 = 0.5* RationalQuadratic(length_scale=1.0, alpha=1.0)

			'''
			k4 = 0.1**2 * RBF(length_scale=0.1) \+ WhiteKernel(noise_level=0.1**2,
						  noise_level_bounds=(1e-3, np.inf))  # noise terms
			'''
			kernel = k1 + k2

		elif kernel == "RationalQuadratic":

			# medium term irregularities
			kernel = RationalQuadratic(length_scale=1.0, alpha=1.0)    

		else:
			kernel = Matern(length_scale=2, nu=3/2)

		self.model = CustomGPR(kernel=kernel)
		return self.model 
