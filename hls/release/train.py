# -*- coding: utf-8 -*-
import os
import pickle
import sys
import argparse

import numpy as np
import xgboost as xgb
from sklearn.linear_model import Lasso


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type = str, default = './data/data_train.pkl', 
                    help = 'Directory to the input dataset. ')

Target_Names = ['LUT', 'FF', 'DSP', 'BRAM']


def load_data(file_name, silence=False):
    if not silence: print "Load data from: ", file_name
    
    if not os.path.exists(file_name):
        sys.exit("Data file " + file_name + " does not exist!")
        
    with open(file_name, "rb") as f:
        data = pickle.load(f)  
        
    return data[0], data[1]


def save_models(models, silence=False):
    if not os.path.exists("./saves/train/"):
        os.mkdir("./saves/train/")
        
    with open("./saves/train/models.pkl", "wb") as f:
        pickle.dump(models, f)
        
    if not silence: print "Models are saved to: ", "./saves/train/models.pkl"
    
    
def get_params_xgb(targetid=None):
    """
    Get the Tuned parameters for xgboost
    """
    param_defaults = []
    
    # for Target 0
    param_defaults.append({'learning_rate': 0.1,
                           'n_estimators': 600,
                           'max_depth': 5,
                           'min_child_weight': 1,
                           'subsample': 1,
                           'colsample_bytree': 1,
                           'gamma':0 })
    
    # for Target 1
    param_defaults.append({'learning_rate': 0.1,
                           'n_estimators': 600,
                           'max_depth': 5,
                           'min_child_weight': 1,
                           'subsample': 1,
                           'colsample_bytree': 1,
                           'gamma':0 })
    
    # for Target 2
    param_defaults.append({'learning_rate': 0.1,
                           'n_estimators': 600,
                           'max_depth': 5,
                           'min_child_weight': 1,
                           'subsample': 1,
                           'colsample_bytree': 1,
                           'gamma':0 })
    
    # for Target 3
    param_defaults.append({'learning_rate': 0.1,
                           'n_estimators': 600,
                           'max_depth': 5,
                           'min_child_weight': 1,
                           'subsample': 1,
                           'colsample_bytree': 1,
                           'gamma':0 })
    
    # return
    if targetid is not None:
        return param_defaults[targetid]
    else:
        return param_defaults
    
    
def get_params_lasso(targetid=None):
    """
    Get the Tuned parameters for xgboost
    """
    param_defaults = []
    
    # for Target 0
    param_defaults.append({'alpha': 25.0})
    
    # for Target 1
    param_defaults.append({'alpha': 85.0})
    
    # for Target 2
    param_defaults.append({'alpha': 2.0})
    
    # for Target 3
    param_defaults.append({'alpha': 2.0})
    
    # return
    if targetid is not None:
        return param_defaults[targetid]
    else:
        return param_defaults
    
    
def train_models(X, Y, silence=False):
    
    models = []
    for ii in xrange(0, 2): 
        print 'Train model for', Target_Names[ii], '...'
    
        params_xgb = get_params_xgb(ii)
        params_lasso = get_params_lasso(ii)
    
        # fix the random seed
        np.random.seed(seed = 100)
        
        # xgboost - feature selection by xgboost
        model_xgb = xgb.XGBRegressor(learning_rate=params_xgb['learning_rate'],
                                     n_estimators=params_xgb['n_estimators'],
                                     max_depth=params_xgb['max_depth'],
                                     min_child_weight=params_xgb['min_child_weight'],
                                     subsample=params_xgb['subsample'],
                                     colsample_bytree=params_xgb['colsample_bytree'],
                                     gamma=params_xgb['gamma'])
        model_xgb.fit(X, Y[:, ii])
    
        b = model_xgb.get_booster()
        feature_weights = [b.get_score(importance_type='weight').get(f, 0.) for f in b.feature_names]
        feature_weights = np.array(feature_weights, dtype=np.float32)
        feature_select  = (feature_weights / feature_weights.sum()) > 0.05
        
        # xgboost - train model
        model_xgb = xgb.XGBRegressor(learning_rate=params_xgb['learning_rate'],
                                     n_estimators=params_xgb['n_estimators'],
                                     max_depth=params_xgb['max_depth'],
                                     min_child_weight=params_xgb['min_child_weight'],
                                     subsample=params_xgb['subsample'],
                                     colsample_bytree=params_xgb['colsample_bytree'],
                                     gamma=params_xgb['gamma'])
        model_xgb.fit(X[:, feature_select], Y[:, ii])
        
        # fix the random seed
        np.random.seed(seed = 100)
        
        # lasso
        model_lasso = Lasso(alpha=params_lasso['alpha'])
        model_lasso.fit(X, Y[:, ii])
        
        # add to list
        models.append([model_xgb, model_lasso, feature_select])
    
    # return
    return models
    

if __name__ == '__main__':
    
    # print info
    print "\n========== Start training models ==========\n"
    
    # parser
    FLAGS, unparsed = parser.parse_known_args()
    
    # load training data
    X, Y = load_data(FLAGS.data_dir)
    
    # train models
    models = train_models(X, Y)
    
    # save models
    save_models(models)

    print "\n========== End ==========\n"