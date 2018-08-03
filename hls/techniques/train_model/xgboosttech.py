import numpy as np

import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor

from .technique import TrainTechniqueBase

class XGBoostTech(TrainTechniqueBase):
    """
    """
    
    def __init__(self, silence=False, def_params_id=-1, 
                 fsel_threshhold=0.03, fsel_round=1, 
                 bagging_ratio=0.8, bagging_round=25, bagging_replace=False):
        
        super(XGBoostTech, self).__init__(silence=silence, name=self.default_name())
        
        self.def_params_id = def_params_id
        self.fsel_threshhold = fsel_threshhold
        self.fsel_round = fsel_round
        self.bagging_ratio = bagging_ratio
        self.bagging_round = bagging_round
        self.bagging_replace = bagging_replace
        
    
    def train(self, X_train, Y_train, params=None, random_seed=None):
        # default parameters
        if params == None:
            params = self.get_def_params()
        
        # super func
        super(XGBoostTech, self).train(X_train, Y_train, params, random_seed=random_seed)
        
        # init feature list
        self.sel_features = np.array([x for x in xrange(self.X_train.shape[1])])
        
        if not self.silence: 
            print '\nSelecting features. fsel_threshhold =', self.fsel_threshhold, '. fsel_round =', self.fsel_round, '\n'
        
        # select features
        for ii in xrange(self.fsel_round):
            # train xgboost
            _model_xgb = xgb.XGBRegressor(learning_rate=params['learning_rate'],
                                          n_estimators=params['n_estimators'],
                                          max_depth=params['max_depth'],
                                          min_child_weight=params['min_child_weight'],
                                          subsample=params['subsample'],
                                          colsample_bytree=params['colsample_bytree'],
                                          gamma=params['gamma'])
            _model_xgb.fit(self.X_train[:, self.sel_features], self.Y_train)
        
            # select features
            b = _model_xgb.get_booster()
            _fweights = [b.get_score(importance_type='weight').get(f, 0.) for f in b.feature_names]
            _fweights = np.array(_fweights, dtype=np.float32)
            _fweights  = _fweights / _fweights.sum()
            _fsel = _fweights > min(0.1, max(self.fsel_threshhold, _fweights.min()))
            self.sel_features = self.sel_features[_fsel]
            # print _fsel
        
        # init model list
        self.model = []
        _X_train = self.X_train[:, self.sel_features]
        
        if not self.silence: 
            print '\nTraining models. bagging_ratio =', self.bagging_ratio, '. bagging_round =', self.bagging_round, '\n'
            
        # train models
        for ii in xrange(self.bagging_round):
            # subsample data
            _sel_index = np.random.choice(self.X_train.shape[0], int(self.X_train.shape[0] * self.bagging_ratio), replace=self.bagging_replace)
            
            # train model
            _model = xgb.XGBRegressor(learning_rate=params['learning_rate'],
                                      n_estimators=params['n_estimators'],
                                      max_depth=params['max_depth'],
                                      min_child_weight=params['min_child_weight'],
                                      subsample=params['subsample'],
                                      colsample_bytree=params['colsample_bytree'],
                                      gamma=params['gamma'])
            _model.fit(_X_train[_sel_index, :], self.Y_train[_sel_index])
            
            # add model to list
            self.model.append(_model)
        
    
    def test(self, X_test, Y_test):
        super(XGBoostTech, self).test(X_test, Y_test)
        
        # test model
        _Y_test_pre = self.predict(self.X_test)
        
        # return score
        return self.score(self.Y_test, _Y_test_pre)
    
    
    def predict(self, X_test):
        super(XGBoostTech, self).predict(X_test)
        
        # predict
        _predict = np.zeros([X_test.shape[0], self.bagging_round])
        for ii in xrange(self.bagging_round):
            _predict[:, ii] = self.model[ii].predict(X_test[:, self.sel_features])
        
        # average and return
        return _predict.mean(axis=1)
    
    
    def get_def_params(self):
        params = {}
        if self.def_params_id == 0: 
            params['learning_rate'] = 0.1
            params['n_estimators'] = 60
            params['max_depth'] = 5
            params['min_child_weight'] = 1
            params['subsample'] = 1
            params['colsample_bytree'] = 1
            params['gamma'] = 0
            
        elif self.def_params_id == 1: 
            params['learning_rate'] = 0.1
            params['n_estimators'] = 60
            params['max_depth'] = 5
            params['min_child_weight'] = 1
            params['subsample'] = 1
            params['colsample_bytree'] = 1
            params['gamma'] = 0
            
        else: 
            params['learning_rate'] = 0.1
            params['n_estimators'] = 60
            params['max_depth'] = 5
            params['min_child_weight'] = 1
            params['subsample'] = 1
            params['colsample_bytree'] = 1
            params['gamma'] = 0
        
        return params