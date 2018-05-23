import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from .technique import TransferTechniqueBase


class WeightSourceTech(TransferTechniqueBase):
    """
    Instances knowledge tranferring technique.
    
    Parameters
    ----------
    max_iteration : int
        Number of training iterations.
        
    n_estimators_ratio : float. (0 to 1)
        Use to decrease the parameter "n_estimators" used in XGBoost.
        
    max_depth_ratio : float. (0 to 1)
        Use to decrease the parameter "max_depth" used in XGBoost.
    """
    
    def __init__(self, se_scale=4, te_scale=6, sb=0.70, tb=0.60): # sb=0.95, tb=0.90
        super(WeightSourceTech, self).__init__()
        self.se_scale = se_scale
        self.te_scale = te_scale
        self.tb = tb
        self.sb = sb
        
    
    def train(self, data_train, data_train_add, params={},):
        super(WeightSourceTech, self).train(data_train, data_train_add)
        
        # data size
        self.data_source_size = self.X_train.shape[0]
        self.data_target_size = self.X_train_add.shape[0]
        self.data_size = self.data_source_size + self.data_target_size
        
        # prepare training data
        _X_train = np.vstack([self.X_train, self.X_train_add.copy()])
        _Y_train = np.hstack([self.Y_train, self.Y_train_add.copy()])
        _weight = np.ones([self.data_size], dtype=np.float64) * 1
        
        # init the model
        # self.model_init()
        
        _mymodel0 = xgb.XGBRegressor(learning_rate=params['learning_rate'],
                                     n_estimators=int(np.ceil(params['n_estimators'])),
                                     max_depth=int(np.ceil(params['max_depth'])),
                                     min_child_weight=params['min_child_weight'],
                                     subsample=params['subsample'],
                                     colsample_bytree=params['colsample_bytree'],
                                     gamma=params['gamma'])
        _mymodel0.fit(self.X_train, self.Y_train)
            
        # train
        print "\nStart training..."
        for ii in xrange(1):
            # print info
            print "Train model. Iteration", ii, ". Parameters are", params, "Weights are", _weight
            
            # train model
            _mymodel = xgb.XGBRegressor(learning_rate=params['learning_rate'],
                                        n_estimators=int(np.ceil(params['n_estimators'])),
                                        max_depth=int(np.ceil(params['max_depth'])),
                                        min_child_weight=params['min_child_weight'],
                                        subsample=params['subsample'],
                                        colsample_bytree=params['colsample_bytree'],
                                        gamma=params['gamma'])
            
            _mymodel.fit(_X_train, _Y_train, sample_weight=_weight)
            
            # cal culate error
            _errors1, _errors2 = self.error_func(_mymodel.predict(_X_train), _mymodel0.predict(_X_train), _Y_train)
        
            # update weight
            # _b = (_errors[self.data_source_size: self.data_source_size + self.data_target_size] \
            #         * _weight[self.data_source_size: self.data_source_size + self.data_target_size]).sum() \
            #         / _weight[self.data_source_size: self.data_source_size + self.data_target_size].sum()
                    
            # _b = _b / (1 - _b)
            
            _bs = self.sb # 1 / (1 + (2 * np.log(self.data_source_size / self.max_iteration)) ** 0.5) # 0.5
            
            _bt = self.tb # 1 / (1 + (2 * np.log(self.data_target_size / self.max_iteration)) ** 0.5) # 0.4
                    
            _weight[0: self.data_source_size] = \
                    _weight[0: self.data_source_size] * _bs ** (_errors1[0: self.data_source_size])
                    
            _weight[self.data_source_size: self.data_size] = \
                    _weight[self.data_source_size: self.data_size] * _bt ** (-_errors2[self.data_source_size: self.data_size])
                    
            _weight = _weight / _weight.sum() * self.data_size
            # _weight[0: self.data_source_size] = \
            #         _weight[0: self.data_source_size] / _weight[0: self.data_source_size].sum() * self.data_source_size
            
            # append model
            # self.model_append(_mymodel, 1 / _b)
            # self.model_append(_mymodel, 1)
        
        # train model
        self.model = xgb.XGBRegressor(learning_rate=params['learning_rate'],
                                      n_estimators=int(np.ceil(params['n_estimators'])),
                                      max_depth=int(np.ceil(params['max_depth'])),
                                      min_child_weight=params['min_child_weight'],
                                      subsample=params['subsample'],
                                      colsample_bytree=params['colsample_bytree'],
                                      gamma=params['gamma'])
            
        self.model.fit(_X_train, _Y_train, sample_weight=_weight)
        
        # figure
        plt.figure()
        plt.plot(_weight)
        plt.show()
        
        # plt.figure()
        # plt.plot(_errors)
        # plt.show()
    
    
    def test(self, data_test):
        super(WeightSourceTech, self).test(data_test)
        
        # test model
        _Y_test_pre = self.model.predict(self.X_test)
        
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
        _start = _end - 1 # _end - self.used_iteration
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
    
    
    def error_func(self, y_pre, y_pre0, y):
        _mean = y.mean()
        _std = y.std()
        
        y = (y - _mean) / _std
        y_pre = (y_pre - _mean) / _std
        y_pre0 = (y_pre0 - _mean) / _std
        
        _errors1 = np.abs(np.abs(y - y_pre) - np.abs(y - y_pre0)) / np.abs(y + np.finfo(float).eps)
        _errors1 = _errors1 / _errors1[0: self.data_source_size].mean() / 80  * self.se_scale
        _errors1 = np.tanh(_errors1) # (np.exp(_errors) - np.exp(-_errors)) / (np.exp(_errors) + np.exp(-_errors))
        
        _errors2 = np.abs(y - y_pre) # / np.abs(y + np.finfo(float).eps)
        _errors2 = _errors2 / _errors2[self.data_source_size: self.data_size].mean() / 100 * self.te_scale
        # _errors2 = _errors2 * self.te_scale * 100
        _errors2 = np.tanh(_errors2)
        
        plt.figure()
        plt.plot(_errors1)
        plt.show()
        
        plt.figure()
        plt.plot(_errors2)
        plt.show()
            
        return _errors1, _errors2
        