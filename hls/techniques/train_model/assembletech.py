import numpy as np

import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor

from technique import TrainTechniqueBase
from technique import TrainTech
from featureselecttech import FeatureSelectTech
from baggingtech import BaggingTech
from xgboosttech import XGBoostTech

class AssembleTech(TrainTechniqueBase):
    """
    """
    
    def __init__(self, model_names, def_params_id=-1):
        super(AssembleTech, self).__init__(name=self.default_name() + ' - ' + str(model_names))
        self.model_names = model_names
        self.def_params_id = def_params_id
        
    
    def train(self, X_train, Y_train, params=None, params_id=0, random_seed=None):
        # default parameters
        if params == None:
            params = self.get_params()
            
        if 'lasso_alpha' not in params: params['lasso_alpha'] = 150
        if 'xgb_learning_rate' not in params: params['xgb_learning_rate'] = 0.1
        if 'xgb_n_estimators' not in params: params['xgb_n_estimators'] = 60
        if 'xgb_max_depth' not in params: params['xgb_max_depth'] = 5
        if 'xgb_min_child_weight' not in params: params['xgb_min_child_weight'] = 1
        if 'xgb_subsample' not in params: params['xgb_subsample'] = 1
        if 'xgb_colsample_bytree' not in params: params['xgb_colsample_bytree'] = 1
        if 'xgb_gamma' not in params: params['xgb_gamma'] = 0
        if 'ann_hidden_layer_sizes' not in params: params['ann_hidden_layer_sizes'] = 0
        
        # super func
        super(AssembleTech, self).train(X_train, Y_train, params, random_seed=random_seed)
        
        # train models
        self.model = []
        for ii in xrange(len(self.model_names)):
            # init model
            _technique = None
            
            if self.model_names[ii] == 'lasso':
                _technique = TrainTech('lasso', silence=True)
                _technique.train(self.X_train, self.Y_train, {'alpha': params['lasso_alpha']})
                
            if self.model_names[ii] == 'lasso_bagging':
                _technique = BaggingTech('lasso', def_params_id=self.def_params_id,
                                         bagging_ratio=0.8, bagging_round=15, silence=True)
                _technique.train(self.X_train, self.Y_train)
                
            elif self.model_names[ii] == 'xgb':
                _technique = TrainTech('xgb', silence=True)
                _technique.train(self.X_train, self.Y_train, 
                                 {'learning_rate': params['xgb_learning_rate'],
                                  'n_estimators': params['xgb_n_estimators'],
                                  'max_depth': params['xgb_max_depth'],
                                  'min_child_weight': params['xgb_min_child_weight'],
                                  'subsample': params['xgb_subsample'],
                                  'colsample_bytree': params['xgb_colsample_bytree'],
                                  'gamma': params['xgb_gamma']})
            
            elif self.model_names[ii] == 'ann':
                _technique = TrainTech('ann', silence=True)
                _technique.train(self.X_train, self.Y_train, {'hidden_layer_sizes': params['ann_hidden_layer_sizes']})
                
            elif self.model_names[ii] == 'xgb_selfs':
                _technique = XGBoostTech(def_params_id=self.def_params_id, 
                                         fsel_threshhold=0.03, fsel_round=1, 
                                         bagging_ratio=1, bagging_round=1, bagging_replace=False)
                _technique.train(self.X_train, self.Y_train)
                
                '''
                _technique = FeatureSelectTech('xgb', sel_method='xgb', sel_threshhold=0.03, silence=True)
                _technique.train(self.X_train, self.Y_train, 
                                 {'learning_rate': params['xgb_learning_rate'],
                                  'n_estimators': params['xgb_n_estimators'],
                                  'max_depth': params['xgb_max_depth'],
                                  'min_child_weight': params['xgb_min_child_weight'],
                                  'subsample': params['xgb_subsample'],
                                  'colsample_bytree': params['xgb_colsample_bytree'],
                                  'gamma': params['xgb_gamma']})
                '''
    
            if self.model_names[ii] == 'xgb_bagging':
                _technique = XGBoostTech(def_params_id=self.def_params_id, 
                                         fsel_threshhold=0, fsel_round=0, 
                                         bagging_ratio=0.8, bagging_round=25, bagging_replace=False)
                
                _technique.train(self.X_train, self.Y_train)
    
            elif self.model_names[ii] == 'xgb_selfs_lasso':
                _technique = FeatureSelectTech('xgb', sel_method='lasso', sel_threshhold=0.4, lasso_alpha=2.0, silence=True)
                _technique.train(self.X_train, self.Y_train, 
                                 {'learning_rate': params['xgb_learning_rate'],
                                  'n_estimators': params['xgb_n_estimators'],
                                  'max_depth': params['xgb_max_depth'],
                                  'min_child_weight': params['xgb_min_child_weight'],
                                  'subsample': params['xgb_subsample'],
                                  'colsample_bytree': params['xgb_colsample_bytree'],
                                  'gamma': params['xgb_gamma']})
    
            elif self.model_names[ii] == 'lasso_selfs_lasso':
                _technique = FeatureSelectTech('lasso', sel_method='lasso', sel_threshhold=0.4, lasso_alpha=2.0, silence=True)
                _technique.train(self.X_train, self.Y_train, {'alpha': 0})
            
            # train model
            if _technique:
                self.model.append(_technique)
                
        
    def test(self, X_test, Y_test):
        super(AssembleTech, self).test(X_test, Y_test)
        
        # test model
        _Y_test_pre = self.predict(self.X_test)
        
        # return score
        return self.score(self.Y_test, _Y_test_pre)
    
    
    
    def predict(self, X_test):
        super(AssembleTech, self).predict(X_test)
        
        # predict
        _Y_test_pre = None
        for ii in xrange(len(self.model)):
            if _Y_test_pre is None:
                _Y_test_pre = self.model[ii].predict(X_test)
            else:
                _Y_test_pre = _Y_test_pre + self.model[ii].predict(X_test)
                
        return _Y_test_pre / len(self.model)
    
    
    def get_params(self):
        params = {}
        if self.def_params_id == 0: 
            params['lasso_alpha'] = 25.0
            
        elif self.def_params_id == 1: 
            params['lasso_alpha'] = 180.0
            
        else:
            params['lasso_alpha'] = 150.0
        
        return params