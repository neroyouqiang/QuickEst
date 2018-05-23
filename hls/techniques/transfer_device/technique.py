# -*- coding: utf-8 -*-
import numpy as np
import xgboost as xgb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from model.model_test import score_REA, score_MSE

class TransferTechniqueBase(object):
    
    def __init__(self, name = None):
        super(TransferTechniqueBase, self).__init__()
        if name:
            self.name = name
        else:
            self.name = self.default_name()
            
        self.model = None
     
        
    def default_name(self):
        return self.__class__.__name__


    def train(self, data_train, data_train_add, target_repeat=1):
        self.X_train = data_train[0].copy()
        self.Y_train = data_train[1].copy()
        
        if target_repeat <= 0:
            self.X_train_add = None
            self.Y_train_add = None
        else:
            self.X_train_add = data_train_add[0].copy()
            self.Y_train_add = data_train_add[1].copy()
            
            for ii in xrange(1, target_repeat):
                self.X_train_add = np.vstack([self.X_train_add, data_train_add[0].copy()])
                self.Y_train_add = np.hstack([self.Y_train_add, data_train_add[1].copy()])
            
        return
    
    
    def test(self, data_test):
        self.X_test = data_test[0]
        self.Y_test = data_test[1]
        return
    
    
    def score(self, Y, Y_pre):
        # get score
        _RAE, errors = score_REA(Y, Y_pre)
        _MSE, errors = score_MSE(Y, Y_pre)
        
        return _RAE, _MSE
    
    
    def train_xgb(self, X_train, Y_train, params=None, weights=None):
        print 'Training model xgb with parameters:', params, 'and data num is', X_train.shape[0], '...'
        
        if params:
            _model = xgb.XGBRegressor(learning_rate=params['learning_rate'],
                                      n_estimators=params['n_estimators'],
                                      max_depth=params['max_depth'],
                                      min_child_weight=params['min_child_weight'],
                                      subsample=params['subsample'],
                                      colsample_bytree=params['colsample_bytree'],
                                      gamma=params['gamma'])
            _model.fit(X_train, Y_train, sample_weight=weights)
        else:
            _model = xgb.XGBRegressor()
            _model.fit(X_train, Y_train)
            
        
        return _model
    
    
    def train_gpr(self, X_train, Y_train, params=None):
        print 'Training model gpr with parameters:', params, 'and data num is', X_train.shape[0], '...'
        
        # Instanciate a Gaussian Process model
        _kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        _model = GaussianProcessRegressor(kernel=_kernel, normalize_y=True, alpha=1e-3)
        
        # Fit to data using Maximum Likelihood Estimation of the parameters
        _model.fit(X_train, Y_train)
        
        # return
        return _model
        

class TrainTwoModelTech(TransferTechniqueBase):
    """
    Simply add two models by ratio.
    """
    
    def __init__(self, ratio=0.5):
        super(TrainTwoModelTech, self).__init__()
        self.ratio = ratio
        
    
    def train(self, data_train, data_train_add, params={}):
        super(TrainTwoModelTech, self).train(data_train, data_train_add)
        
        # data size
        self.data_source_size = self.X_train.shape[0]
        self.data_target_size = self.X_train_add.shape[0]
        self.data_size = self.data_source_size + self.data_target_size
        
        # train
        print "\nStart training..."
        
        # print info
        print "Parameters are", params
            
        # train model
        self.model = {}
        self.model['model1'] = xgb.XGBRegressor(learning_rate=params['learning_rate'],
                                                n_estimators=params['n_estimators'],
                                                max_depth=params['max_depth'],
                                                min_child_weight=params['min_child_weight'],
                                                subsample=params['subsample'],
                                                colsample_bytree=params['colsample_bytree'],
                                                gamma=params['gamma'])
        self.model['model2'] = xgb.XGBRegressor()
            
        self.model['model1'].fit(self.X_train, self.Y_train)
        self.model['model2'].fit(self.X_train_add, self.Y_train_add)
        
    
    
    def test(self, data_test):
        super(TrainTwoModelTech, self).test(data_test)
        
        print (1.0 - self.ratio)
        print (self.ratio)
        # test model
        _Y_test_pre = (1.0 - self.ratio) * self.model['model1'].predict(self.X_test) + self.ratio * self.model['model2'].predict(self.X_test)
        
        # return score
        return self.score(self.Y_test, _Y_test_pre)
    

class TrainDataTogetherTech(TransferTechniqueBase):
    """
    Simply put the target and source training data together as the whole training data and train models
    
    Parameters
    ----------
    target_repeat : int
        How many times the target training data is repeated in the whole training data.
    
    model_name : string
        The used model.
        'xgb' - XGBoost
    """
    
    def __init__(self, target_repeat=1, model_name='xgb'):
        super(TrainDataTogetherTech, self).__init__()
        self.target_repeat = target_repeat
        self.model_name = model_name
    
    
    def train(self, data_train, data_train_add, params={}, model_name="xgb"):
        super(TrainDataTogetherTech, self).train(data_train, data_train_add, target_repeat=self.target_repeat)
        
        # prepare training data
        if self.X_train_add is None:
            _X_train = self.X_train
            _Y_train = self.Y_train
        else:
            _X_train = np.vstack([self.X_train, self.X_train_add])
            _Y_train = np.hstack([self.Y_train, self.Y_train_add])
        
        # train model
        if self.model_name == 'xgb':
            self.model = self.train_xgb(_X_train, _Y_train, params)
        elif self.model_name == 'gpr':
            self.model = self.train_gpr(_X_train, _Y_train, params)
    
    
    def test(self, data_test):
        super(TrainDataTogetherTech, self).test(data_test)
        
        # test model
        _Y_test_pre = self.model.predict(self.X_test)
        
        # return score
        return self.score(self.Y_test, _Y_test_pre)
        