import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from .technique import TransferTechniqueBase


class RmvMisleadTech(TransferTechniqueBase):
    """
    Instances knowledge tranferring technique. Remove misleading source training data.
    Reference: Instance weighting for domain adaptation in NLP, 2007.
        
    Parameters
    ----------
    target_repeat : int
        How many times the target training data is repeated in the whole training data. Have the same effect as the paramter lamda_t_l in the paper.
    """
    
    def __init__(self, rmv_num=4, target_repeat=1):
        super(RmvMisleadTech, self).__init__()
        self.target_repeat = target_repeat
        self.rmv_num = rmv_num
        
    
    def train(self, data_train, data_train_add, params={},):
        super(RmvMisleadTech, self).train(data_train, data_train_add)
        
        # train model on target training data
        _model_rmv = self.train_xgb(self.X_train_add, self.Y_train_add)
        
        _error = np.abs(_model_rmv.predict(self.X_train) - self.Y_train) / (self.Y_train + np.finfo(float).eps)
        
        _index_remain = np.argsort(_error)
        _index_remain = _index_remain[0:-self.rmv_num]
        
        # prepare training data
        _X_train = self.X_train[_index_remain].copy()
        _Y_train = self.Y_train[_index_remain].copy()
        
        for ii in xrange(self.target_repeat):
            _X_train = np.vstack([_X_train, self.X_train_add.copy()])
            _Y_train = np.hstack([_Y_train, self.Y_train_add.copy()])
            
        # train model
        self.model = self.train_xgb(_X_train, _Y_train, params)
    
    
    def test(self, data_test):
        super(RmvMisleadTech, self).test(data_test)
        
        # test model
        _Y_test_pre = self.model.predict(self.X_test)
        
        # return score
        return self.score(self.Y_test, _Y_test_pre)
    

