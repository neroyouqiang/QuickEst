# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from .technique import TransferTechniqueBase


class FeatureCopyTech(TransferTechniqueBase):
    """
    Reference: Frustratingly Easy Domain Adaptation, 2007.
        
    Parameters
    ----------
    """
    
    def __init__(self):
        super(FeatureCopyTech, self).__init__(target_repeat=1)
        
    
    def train(self, data_train, data_train_add, params={},):
        super(FeatureCopyTech, self).train(data_train, data_train_add)
        
        # copy features
        _X_train_source = np.hstack([self.X_train.copy(), np.ones(self.X_train.shape) * np.nan]) # , self.X_train.copy()
        _X_train_target = np.hstack([self.X_train_add.copy(), self.X_train_add.copy()]) # , np.ones(self.X_train_add.shape) * np.nan
        _Y_train_source = self.Y_train.copy()
        _Y_train_target = self.Y_train_add.copy()
        
        # prepare training data
        _X_train = _X_train_source
        _Y_train = _Y_train_source
        
        for ii in xrange(self.target_repeat):
            _X_train = np.vstack([_X_train, _X_train_target.copy()])
            _Y_train = np.hstack([_Y_train, _Y_train_target.copy()])
            
        # train model
        # params['max_depth'] = params['max_depth'] + 2
        self.model = self.train_xgb(_X_train, _Y_train, params)
        
        # print info
        feature_order = np.argsort(self.model.feature_importances_)
        print feature_order
    
    def test(self, data_test):
        super(FeatureCopyTech, self).test(data_test)
        
        # copy features
        _X_test_target = np.hstack([self.X_test.copy(), self.X_test.copy()]) # , np.ones(self.X_test.shape) * np.nan
        
        # test model
        _Y_test_pre = self.model.predict(_X_test_target)
        
        # return score
        return self.score(self.Y_test, _Y_test_pre)
    


