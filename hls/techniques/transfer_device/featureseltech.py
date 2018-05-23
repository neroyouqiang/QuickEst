# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import numpy as np
import pickle
import matplotlib.pyplot as plt
import xgboost as xgb

from .technique import TransferTechniqueBase


class FeatureSelTech(TransferTechniqueBase):
    """
    Select part of the features to train new models.
    Reference: 
        
    Parameters
    ----------
    """
    
    def __init__(self, target_repeat=1, threshhold=0.995, select_time=7):
        super(FeatureSelTech, self).__init__()
        
        self.init(target_repeat=target_repeat, 
                  threshhold=threshhold, 
                  select_time=select_time)
        
        
    def init(self, target_repeat=1, threshhold=0.995, select_time=7):
        self.target_repeat = target_repeat
        self.threshhold = threshhold
        self.select_time = select_time
        
        self.selfeatures = None
        
    
    def train(self, data_train, data_train_add, params={}, params_fsel=None):
        super(FeatureSelTech, self).train(data_train, data_train_add, target_repeat=self.target_repeat)
        # handle input
        if params_fsel is None:
            params_fsel = params
        
        # select features
        self.selfeatures = self.get_important_features(params_fsel)
        print "Selected features are", self.selfeatures
        
            
        # train model
        _X_train = np.vstack([self.X_train[:, self.selfeatures], self.X_train_add[:, self.selfeatures]])
        _Y_train = np.hstack([self.Y_train, self.Y_train_add])
        self.model = self.train_xgb(_X_train, _Y_train, params)
    
    
    def test(self, data_test):
        super(FeatureSelTech, self).test(data_test)
        
        # select features
        _X_test = self.X_test[:, self.selfeatures]
        
        # test model
        _Y_test_pre = self.model.predict(_X_test)
        
        # return score
        return self.score(self.Y_test, _Y_test_pre)
    

    def get_important_features(self, params):
        """
        Feature selection strategy. It can be rewritten to change the strategy.
        """
        
        # testmodel = pickle.load(open("saves/models/xgboost_target0_v2", "r"))
        
        _X_train = np.vstack([self.X_train, self.X_train_add])
        _Y_train = np.hstack([self.Y_train, self.Y_train_add])
        _orders = np.array([ii for ii in xrange(_X_train.shape[1])])
        
        for ii in xrange(self.select_time):
            testmodel = self.train_xgb(_X_train[:, _orders], _Y_train, params)
            _orders = _orders[self.get_xgb_important_features(testmodel)]
        
        # return np.array([21, 3, 56, 22, 19, 13, 2, 0, 1, 49, 24, 72, 43])
        return _orders

    def get_xgb_important_features(self, model):
        
        # the most important features
        b = model.get_booster()
        
        fs = b.get_score(importance_type='weight') # importance_type='weight' gain
        all_features = [fs.get(f, 0.) for f in b.feature_names]
        all_features = np.array(all_features, dtype=np.float32)
        
        feature_importance_g = all_features / all_features.sum()
        
        # fs = b.get_score(importance_type='weight') # importance_type='weight'
        # all_features = [fs.get(f, 0.) for f in b.feature_names]
        # all_features = np.array(all_features, dtype=np.float32)
        
        # feature_importance_w = all_features / all_features.sum()
        
        # feature_importance = (feature_importance_g * 3 + feature_importance_w * 0) / 3
        feature_importance = feature_importance_g
        
        feature_order = np.argsort(feature_importance) 
        feature_order = feature_order[::-1]
        
        feature_importance_sorted = feature_importance[feature_order]
        feature_importance_cumsum = feature_importance_sorted.cumsum()
        selected_num = feature_importance_cumsum[feature_importance_cumsum<self.threshhold] # 0.95
        # selected_num = feature_importance_cumsum[feature_importance_sorted>0.01] # 0.01
        
        print feature_importance_sorted[0: np.max([selected_num.shape[0], 10])]
        
        # plt.figure()
        # plt.plot(feature_importance_cumsum)
        # plt.show()
        
        return feature_order[0: np.max([selected_num.shape[0], 10])] # , feature_order.shape[0] - 5