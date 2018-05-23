import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from .technique import TransferTechniqueBase


class TrAdaboostTech(TransferTechniqueBase):
    """
    Instances knowledge tranferring technique. Change the training data weight based on the error.
    Reference: Boosting for Transfer Learning, 2007.
    
    Parameters
    ----------
    max_iteration : int
        Number of training iterations.
        
    n_estimators_ratio : float. (0 to 1)
        Use to decrease the parameter "n_estimators" used in XGBoost.
        
    max_depth_ratio : float. (0 to 1)
        Use to decrease the parameter "max_depth" used in XGBoost.
    """
    
    def __init__(self, max_iteration=6, n_estimators_ratio=0.8, max_depth_ratio=0.8):
        super(TrAdaboostTech, self).__init__()
        
        self.init(max_iteration=max_iteration, 
                  n_estimators_ratio=n_estimators_ratio, 
                  max_depth_ratio=max_depth_ratio)
        
        
    def init(self, max_iteration=6, n_estimators_ratio=0.8, max_depth_ratio=0.8):
        self.max_iteration = max_iteration
        self.n_estimators_ratio = n_estimators_ratio
        self.max_depth_ratio = max_depth_ratio
         
    
    def train(self, data_train, data_train_add, params={},):
        super(TrAdaboostTech, self).train(data_train, data_train_add)
        
        # data size
        self.data_source_size = self.X_train.shape[0]
        self.data_target_size = self.X_train_add.shape[0]
        self.data_size = self.data_source_size + self.data_target_size
        
        # prepare training data
        _X_train = np.vstack([self.X_train, self.X_train_add.copy()])
        _Y_train = np.hstack([self.Y_train, self.Y_train_add.copy()])
        _weight = np.ones([self.data_size], dtype=np.float64) * 1
        
        # init the model
        self.model_init()
        
        # train
        print "\nStart training by TrAdaBoost..."
        for ii in xrange(self.max_iteration):
            # print info
            print "Train model. Iteration", ii, ". Parameters are", params, "Weights are", _weight
            
            # train model
            _mymodel = xgb.XGBRegressor(learning_rate=params['learning_rate'],
                                          n_estimators=int(np.ceil(params['n_estimators'] * self.n_estimators_ratio)),
                                          max_depth=int(np.ceil(params['max_depth'] * self.max_depth_ratio)),
                                          min_child_weight=params['min_child_weight'],
                                          subsample=params['subsample'],
                                          colsample_bytree=params['colsample_bytree'],
                                          gamma=params['gamma'])
            
            _mymodel.fit(_X_train, _Y_train, sample_weight=_weight)
            
            # cal culate error
            _errors = self.error_func(_mymodel.predict(_X_train), _Y_train)
        
            # update weight
            _b = (_errors[self.data_source_size: self.data_source_size + self.data_target_size] \
                    * _weight[self.data_source_size: self.data_source_size + self.data_target_size]).sum() \
                    / _weight[self.data_source_size: self.data_source_size + self.data_target_size].sum()
                    
            _b = _b / (1 - _b)
            
            _bs = 1 / (1 + (2 * np.log(self.data_source_size / self.max_iteration)) ** 0.5)
            
            # _bt = 1 / (1 + (2 * np.log(self.data_target_size / self.max_iteration)) ** 0.5)
            
            # print "_bt =", _bt
            # print "_bs =", _bs
                    
            _weight[0: self.data_source_size] = \
                    _weight[0: self.data_source_size] * _bs ** _errors[0: self.data_source_size]
                    
            _weight[self.data_source_size: self.data_size] = \
                    _weight[self.data_source_size: self.data_size] * _b ** (- _errors[self.data_source_size: self.data_size])
                    
            _weight = _weight / _weight.sum() * self.data_size
            
            # append model
            self.model_append(_mymodel, 1 / _b)
            
            # plt.figure()
            # plt.plot(_weight)
            # plt.show()
        
            # plt.figure()
            # plt.plot(_errors)
            # plt.show()
    
    
    def test(self, data_test):
        super(TrAdaboostTech, self).test(data_test)
        
        # test model
        _Y_test_pre = self.model_predict(self.X_test)
        
        # return score
        return self.score(self.Y_test, _Y_test_pre)
    
    
    def model_init(self):
        self.model = {}
        self.model['models'] = []
        self.model['errors'] = []
        
        
    def model_append(self, mymodel, error=1):
        self.model['models'].append(mymodel)
        self.model['errors'].append(error)
        
    
    def model_predict(self, X):
        _mymodels = self.model['models']
        _myerrors = self.model['errors']
        print _myerrors
        
        # index of the used iterations
        _end = len(_mymodels)
        _start = 0 # _end - self.used_iteration
        if _start < 0: _start = 0
        
        # predict
        Y_pre = np.zeros([X.shape[0]], dtype=np.float64)
        error_sum = 0
        for ii in xrange(_start, _end):
            Y_pre = Y_pre + _mymodels[ii].predict(X) * _myerrors[ii]
            error_sum = error_sum + _myerrors[ii]
        Y_pre = Y_pre  / error_sum
        
        # return
        return Y_pre
    
    
    def error_func(self, y_pre, y):
        _errors = np.abs(y - y_pre)
        _errors = _errors / _errors.max()
        
        return _errors
        