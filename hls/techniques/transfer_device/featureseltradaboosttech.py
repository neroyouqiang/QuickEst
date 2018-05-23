# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import numpy as np
import pickle
import matplotlib.pyplot as plt
import xgboost as xgb

from .technique import TransferTechniqueBase
from .tradaboosttech import TrAdaboostTech
from .featureseltech import FeatureSelTech


class FeatureSelTrAdaboostTech(FeatureSelTech, TrAdaboostTech, TransferTechniqueBase):
    """
    Combine FeatureSelTech and TrAdaboostTech
    Reference: 
        
    Parameters
    ----------
    """
    
    def __init__(self, fsel_threshhold=0.995, fsel_select_time=7, tra_max_iteration=6):
        TransferTechniqueBase.__init__(self)
        TrAdaboostTech.init(self, max_iteration=tra_max_iteration)
        FeatureSelTech.init(self, threshhold=fsel_threshhold, select_time=fsel_select_time)
        
    
    def train(self, data_train, data_train_add, params={}, params_fsel=None):
        TransferTechniqueBase.train(self, data_train, data_train_add)
        
        # handle input
        if params_fsel is None:
            params_fsel = params
        
        # select features
        self.selfeatures = FeatureSelTech.get_important_features(self, params_fsel)
        print "Selected features are", self.selfeatures
        
        _X_train_source = self.X_train[:, self.selfeatures]
        _X_train_target = self.X_train_add[:, self.selfeatures]
        
        # train model
        TrAdaboostTech.train(self, [_X_train_source, self.Y_train], 
                             [_X_train_target, self.Y_train_add], params)
    
    
    def test(self, data_test):
        TransferTechniqueBase.test(self, data_test)
        
        # select features
        _X_test = self.X_test[:, self.selfeatures]
    
        # test model
        _result = TrAdaboostTech.test(self, [_X_test, self.Y_test])
        
        return _result
    