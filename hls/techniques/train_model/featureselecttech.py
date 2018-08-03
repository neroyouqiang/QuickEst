import numpy as np

import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor

from .technique import TrainTechniqueBase

class FeatureSelectTech(TrainTechniqueBase):
    """
    """
    
    def __init__(self, model_name, sel_method='lasso', sel_threshhold=0, lasso_alpha=1, silence=False):
        super(FeatureSelectTech, self).__init__(silence=silence, name=self.default_name() + ' - ' + model_name + ' feature selected by ' + sel_method)
        self.model_name = model_name
        self.sel_method = sel_method
        self.sel_threshhold = sel_threshhold
        self.lasso_alpha = lasso_alpha
        
    
    def train(self, X_train, Y_train, params={}, random_seed=None, params_id=0):
        # default parameters
        if 'alpha' not in params: params['alpha'] = 0.1
        if 'learning_rate' not in params: params['learning_rate'] = 0.1
        if 'n_estimators' not in params: params['n_estimators'] = 600
        if 'max_depth' not in params: params['max_depth'] = 5
        if 'min_child_weight' not in params: params['min_child_weight'] = 1
        if 'subsample' not in params: params['subsample'] = 1
        if 'colsample_bytree' not in params: params['colsample_bytree'] = 1
        if 'gamma' not in params: params['gamma'] = 0
        
        # super func
        super(FeatureSelectTech, self).train(X_train, Y_train, params, random_seed=random_seed)
        
        # select feature
        if self.sel_method == 'lasso':
            # by lasso
            _model_lasso = Lasso(alpha=self.lasso_alpha)
            _model_lasso.fit(self.X_train, self.Y_train)
            self.feature_select = np.abs(_model_lasso.coef_) / np.abs(_model_lasso.coef_).sum() > self.sel_threshhold
            
        elif self.sel_method == 'xgb':
            # by xgboost
            _model_xgb = xgb.XGBRegressor(learning_rate=params['learning_rate'],
                                          n_estimators=params['n_estimators'],
                                          max_depth=params['max_depth'],
                                          min_child_weight=params['min_child_weight'],
                                          subsample=params['subsample'],
                                          colsample_bytree=params['colsample_bytree'],
                                          gamma=params['gamma'])
            _model_xgb.fit(self.X_train, self.Y_train)
        
            b = _model_xgb.get_booster()
            feature_weights = [b.get_score(importance_type='weight').get(f, 0.) for f in b.feature_names]
            feature_weights = np.array(feature_weights, dtype=np.float32)
            self.feature_select  = (feature_weights / feature_weights.sum()) > self.sel_threshhold
          
        # print self.feature_select
            
        # train model
        if self.model_name == 'lasso':
            # by lasso
            self.model = Lasso(alpha=params['alpha'])
            self.model.fit(self.X_train[:, self.feature_select], self.Y_train)
            # print self.model.coef_
            
        elif self.model_name == 'xgb':
            # by xgboost
            self.model = xgb.XGBRegressor(learning_rate=params['learning_rate'],
                                          n_estimators=params['n_estimators'],
                                          max_depth=params['max_depth'],
                                          min_child_weight=params['min_child_weight'],
                                          subsample=params['subsample'],
                                          colsample_bytree=params['colsample_bytree'],
                                          gamma=params['gamma'])
            self.model.fit(self.X_train[:, self.feature_select], self.Y_train)
            
        # print self.model.coef_
        
    
    def test(self, X_test, Y_test):
        super(FeatureSelectTech, self).test(X_test, Y_test)
        
        # test model
        _Y_test_pre = self.predict(self.X_test)
        
        # return score
        return self.score(self.Y_test, _Y_test_pre)
    
    
    def predict(self, X_test):
        super(FeatureSelectTech, self).predict(X_test)
        
        # predict
        return self.model.predict(X_test[:, self.feature_select])
        
    
    """
    def select_features(model, model_name):
        
        if model_name == 'lasso':
            return np.abs(model.coef_) > self.sel_threshhold
        elif method == 'xgb':
            
            feature_importance = feature_importance_g
            
            feature_order = np.argsort(feature_importance) 
            feature_order = feature_order[::-1]
            
            feature_importance_sorted = feature_importance[feature_order]
            feature_importance_cumsum = feature_importance_sorted.cumsum()
            selected_num = feature_importance_cumsum[feature_importance_cumsum<self.threshhold] # 0.95
            # selected_num = feature_importance_cumsum[feature_importance_sorted>0.01] # 0.01
            
            print feature_importance_sorted[0: np.max([selected_num.shape[0], 10])]
    """