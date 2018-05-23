# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import metrics
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from model.model_test2 import score_REA, score_MSE
from model.model_train import model_training

class TrainTechniqueBase(object):
    
    def __init__(self, name = None, silence=False):
        super(TrainTechniqueBase, self).__init__()
        if name:
            self.name = name
        else:
            self.name = self.default_name()
        
        self.silence = silence
        self.model = None
     
        
    def default_name(self):
        return self.__class__.__name__


    def train(self, X_train, Y_train, params={}, random_seed=None):
        self.X_train = X_train.copy()
        self.Y_train = Y_train.copy()
        
        # fix training process
        if random_seed is not None:
            np.random.seed(seed = random_seed)
            
        # print info
        if not self.silence: print "Start training", self.name, "..."
        if not self.silence: print "Parameters are", params, "\n"
        
        return
    
    
    def test(self, X_test, Y_test):
        self.X_test = X_test.copy()
        self.Y_test = Y_test.copy()
        return
    
    
    def score(self, Y, Y_pre, is_plot_errors=False):
        # get score
        _RAE, errors = score_REA(Y, Y_pre)
        _MSE, errors = score_MSE(Y, Y_pre)
        _R2 = metrics.r2_score(Y, Y_pre)
        
        if is_plot_errors:
            plt.figure()
            plt.plot(errors)
            plt.show()
        
        return _RAE, _MSE, _R2
    
    
    def predict(self, X_test):
        return
        

class TrainTech(TrainTechniqueBase):
    """
    Train one basic model.
    """
    
    def __init__(self, model_name):
        super(TrainTech, self).__init__(name=self.default_name() + ' - ' + model_name)
        
        self.model_name = model_name
        
    
    def train(self, X_train, Y_train, params={}, random_seed=None):
        super(TrainTech, self).train(X_train, Y_train, params, random_seed=random_seed)
            
        # train model
        self.model = model_training(self.X_train, self.Y_train, self.model_name, 
                                    hyperparams=params, silence=True)
        
    
    def test(self, X_test, Y_test):
        super(TrainTech, self).test(X_test, Y_test)
        
        # test model
        _Y_test_pre = self.predict(self.X_test)
        
        # return score
        return self.score(self.Y_test, _Y_test_pre, is_plot_errors=False)
    
    
    def predict(self, X_test):
        super(TrainTech, self).predict(X_test)
        
        # predict
        return self.model.predict(X_test)
    
