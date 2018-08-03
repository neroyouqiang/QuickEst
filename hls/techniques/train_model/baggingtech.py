import numpy as np

import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor

from .technique import TrainTechniqueBase

class BaggingTech(TrainTechniqueBase):
    """
    """
    
    def __init__(self, model_name, bagging_ratio=0.8, bagging_round=10, silence=False, def_params_id=-1):
        super(BaggingTech, self).__init__(silence=silence, name=self.default_name() + ' - ' + model_name)
        self.def_params_id = def_params_id
        self.model_name = model_name
        self.bagging_ratio = bagging_ratio
        self.bagging_round = bagging_round
        
    
    def train(self, X_train, Y_train, params=None, random_seed=None):
        # default parameters
        if params == None:
            params = self.get_params()
        
        # super func
        super(BaggingTech, self).train(X_train, Y_train, params, random_seed=random_seed)
        
        # init model list
        self.model = []
            
        # train model
        for ii in xrange(self.bagging_round):
            # subsample data
            _sel_index = np.random.choice(self.X_train.shape[0], int(self.X_train.shape[0] * self.bagging_ratio))
            
            # train model
            if self.model_name == 'lasso':
                # by lasso
                _model = Lasso(alpha=params['lasso_alpha'])
                _model.fit(self.X_train[_sel_index, :], self.Y_train[_sel_index])
            
            elif self.model_name == 'xgb':
                # by xgboost
                _model = xgb.XGBRegressor(learning_rate=params['learning_rate'],
                                              n_estimators=params['n_estimators'],
                                              max_depth=params['max_depth'],
                                              min_child_weight=params['min_child_weight'],
                                              subsample=params['subsample'],
                                              colsample_bytree=params['colsample_bytree'],
                                              gamma=params['gamma'])
                _model.fit(self.X_train[_sel_index, :], self.Y_train[_sel_index])
            
            # add model to list
            self.model.append(_model)
        
    
    def test(self, X_test, Y_test):
        super(BaggingTech, self).test(X_test, Y_test)
        
        # test model
        _Y_test_pre = self.predict(self.X_test)
        
        # return score
        return self.score(self.Y_test, _Y_test_pre)
    
    
    def predict(self, X_test):
        super(BaggingTech, self).predict(X_test)
        
        # predict
        _predict = np.zeros([X_test.shape[0], self.bagging_round])
        for ii in xrange(self.bagging_round):
            _predict[:, ii] = self.model[ii].predict(X_test)
        
        # average and return
        return _predict.mean(axis=1)
    
    
    def get_params(self):
        params = {}
        if self.def_params_id == 0: 
            params['lasso_alpha'] = 20.0
            params['learning_rate'] = 0.1
            params['n_estimators'] = 60
            params['max_depth'] = 5
            params['min_child_weight'] = 1
            params['subsample'] = 1
            params['colsample_bytree'] = 1
            params['gamma'] = 0
            
        elif self.def_params_id == 1: 
            params['lasso_alpha'] = 180.0
            params['learning_rate'] = 0.1
            params['n_estimators'] = 60
            params['max_depth'] = 5
            params['min_child_weight'] = 1
            params['subsample'] = 1
            params['colsample_bytree'] = 1
            params['gamma'] = 0
            
        else: 
            params['lasso_alpha'] = 150.0
            params['learning_rate'] = 0.1
            params['n_estimators'] = 60
            params['max_depth'] = 5
            params['min_child_weight'] = 1
            params['subsample'] = 1
            params['colsample_bytree'] = 1
            params['gamma'] = 0
        
        return params