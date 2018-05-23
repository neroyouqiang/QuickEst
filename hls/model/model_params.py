# -*- coding: utf-8 -*-

def params_xgb_tune_list_n_estimators(param):
    """
    parameter list for tunning n_estimators of xgboost model
    """
    param_list = []
    for x in xrange(10, 200, 10):
        tmp = param.copy()
        tmp['n_estimators'] = x
        param_list.append(tmp)
    
    for x in xrange(200, 1500, 20):
        tmp = param.copy()
        tmp['n_estimators'] = x
        param_list.append(tmp)
    
    return param_list


def params_xgb_tune_list_depth_child_weight(param):
    """
    parameter list for tunning max_depth and min_child_weight of xgboost model
    """
    param_list = []
    for x in xrange(1, 11, 1):
        for y in xrange(1, 11, 1):
            tmp = param.copy()
            tmp['max_depth'] = x
            tmp['min_child_weight'] = y
            param_list.append(tmp)
    
    return param_list


def params_xgb_tune_list_subsample(param):
    """
    parameter list for tunning subsample of xgboost model
    """
    param_list = []
    for x in xrange(25, 51, 1):
        tmp = param.copy()
        tmp['subsample'] = x / 50.0 
        param_list.append(tmp)
    
    return param_list


def params_xgb_tune_list_colsample_bytree(param):
    """
    parameter list for tuning colsample_bytree of xgboost model
    """
    param_list = []
    for x in xrange(25, 51, 1):
        tmp = param.copy()
        tmp['colsample_bytree'] = x / 50.0
        param_list.append(tmp)
    
    return param_list


def params_xgb_tune_list_colsample_subsample(param):
    """
    parameter list for tuning colsample_bytree and subsample of xgboost model
    """
    param_list = []
    for x in xrange(25, 51, 1):
        for y in xrange(25, 51, 1):
            tmp = param.copy()
            tmp['colsample_bytree'] = x / 50.0
            tmp['subsample'] = y / 50.0 
            param_list.append(tmp)
    
    return param_list
    

def params_xgb_tune_list_gamma(param):
    """
    parameter list for tuning gamma of xgboost model
    """
    param_list = []
    for x in xrange(0, 6, 1):
        tmp = param.copy()
        tmp['gamma'] = x / 100.0
        param_list.append(tmp)
    
    return param_list


def params_xgb_v1(targetid=None):
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
    param_defaults.append({'learning_rate':0.1,
                           'n_estimators': 500,
                           'max_depth':5,
                           'min_child_weight':1,
                           'subsample': 1,
                           'colsample_bytree': 1,
                           'gamma':0})
    
    # for Target 2
    param_defaults.append({'learning_rate':0.1,
                           'n_estimators': 400,
                           'max_depth':5,
                           'min_child_weight':1,
                           'subsample': 1,
                           'colsample_bytree': 1,
                           'gamma':0})
    
    # for Target 3
    param_defaults.append({'learning_rate':0.1,
                           'n_estimators': 300,
                           'max_depth':4,
                           'min_child_weight':1,
                           'subsample': 1,
                           'colsample_bytree': 1,
                           'gamma':0})
    
    # return
    if targetid is not None:
        return param_defaults[targetid]
    else:
        return param_defaults
    
    
def params_xgb_v2(targetid=None):
    """
    Get the Tuned parameters for xgboost
    """
    param_defaults = []
    
    # for Target 0
    param_defaults.append({'learning_rate': 0.05,
                           'n_estimators': 1500,
                           'max_depth':4,
                           'min_child_weight':2,
                           'subsample': 1, 
                           'colsample_bytree': 1,
                           'gamma':0})
    
    # for Target 1
    """
    param_defaults.append({'learning_rate': 0.05,
                           'n_estimators': 1000,# 1920,
                           'max_depth': 9,
                           'min_child_weight': 3, # 3,
                           'subsample': 0.8, # 0.92,
                           'colsample_bytree': 0.7,
                           'gamma': 0})
    """
    
    param_defaults.append({'learning_rate': 0.05,
                           'n_estimators': 1000, # 1000
                           'max_depth': 11, 
                           'min_child_weight': 4,
                           'subsample': 0.9, 
                           'colsample_bytree': 0.8, 
                           'gamma': 0, }) # 0
    
    """
    param_defaults.append({'learning_rate': 0.05,
                           'n_estimators': 1000,
                           'max_depth': 8, 
                           'min_child_weight': 2,
                           'subsample': 0.7, 
                           'colsample_bytree': 1, 
                           'gamma': 0 })
    """
    
    # for Target 2
    param_defaults.append({'learning_rate': 0.05,
                           'n_estimators': 500, # 920,
                           'max_depth': 7,
                           'min_child_weight': 1,
                           'subsample': 1, # 0.96,???
                           'colsample_bytree': 1, # 0.94,???
                           'gamma': 0})
    
    # for Target 3
    param_defaults.append({'learning_rate': 0.1,
                           'n_estimators': 300, # 780,
                           'max_depth': 9,
                           'min_child_weight': 1,
                           'subsample': 1,
                           'colsample_bytree': 1,
                           'gamma': 0})
    
    # return
    if targetid is not None:
        return param_defaults[targetid]
    else:
        return param_defaults
    
    
def params_xgb_timing_tune_list_v1(param):
    param_list = []
    for ii in range(5):
        for jj in range(5):
            tmp = param.copy()
            tmp['max_depth'] = ii + 1
            tmp['n_estimators'] = (jj + 1) * 100
            param_list.append(tmp)

    return param_list
    

def params_xgb_timing_v1():
    """
    Get the Tuned parameters for xgboost
    """
    param_defaults = []
    
    # for Target 0
    param_defaults.append({'learning_rate': 0.1,
                           'n_estimators': 200,
                           'max_depth':3,
                           'min_child_weight':1,
                           'subsample': 1, 
                           'colsample_bytree': 1,
                           'gamma':0})
    
    return param_defaults[0]

    
def params_xgb_timing_v2():
    """
    Get the Tuned parameters for xgboost
    """
    param_defaults = []
    
    
    """# by opentuner
    param_defaults.append({'learning_rate': 0.05,
                           'n_estimators': 2000,
                           'max_depth':5,
                           'min_child_weight':2,
                           'subsample': 0.9, 
                           'colsample_bytree': 0.6,
                           'gamma':0.3})
    """
    
    # atuo
    param_defaults.append({'learning_rate': 0.05,
                           'n_estimators': 1500,
                           'max_depth': 9,
                           'min_child_weight':3,
                           'subsample': 0.84,
                           'colsample_bytree': 0.6,
                           'gamma':0.05}) # 0.05
    
    
    return param_defaults[0]


def params_xgb_timing_v3():
    """
    Get the Tuned parameters for xgboost
    """
    param_defaults = []
    
    # manually
    param_defaults.append({'learning_rate': 0.05,
                           'n_estimators': 5000,
                           'max_depth':4, 
                           'min_child_weight':3,
                           'subsample': 0.3,
                           'colsample_bytree': 0.3,
                           'gamma':0.8})
    
    return param_defaults[0]    

    
def params_linxgb(targetid=None):
    """
    Get the Tuned parameters for xgboost
    """
    param_defaults = []
    
    # for Target 0
    param_defaults.append({'learning_rate': 0.05,
                           'n_estimators': 50,
                           'max_depth':4,
                           'min_samples_leaf':2,
                           'subsample': 1, 
                           'gamma':0})
    
    # for Target 1
    param_defaults.append({'learning_rate': 0.05,
                           'n_estimators': 1000,
                           'max_depth': 9,
                           'min_samples_leaf': 3,
                           'subsample': 0.8,
                           'gamma': 0})
    
    # for Target 2
    param_defaults.append({'learning_rate': 0.05,
                           'n_estimators': 500,
                           'max_depth': 7,
                           'min_samples_leaf': 1,
                           'subsample': 1,
                           'gamma': 0})
    
    # for Target 3
    param_defaults.append({'learning_rate': 0.1,
                           'n_estimators': 100,
                           'max_depth': 9,
                           'min_samples_leaf': 1,
                           'subsample': 1, 
                           'gamma': 0})
    
    # return
    if targetid is not None:
        return param_defaults[targetid]
    else:
        return param_defaults
    
    
    
def params_cb_tune_iterations(param):
    """
    parameter list for tunning iterations of catboost model
    """
    param_list = []
    for ii in xrange(100, 1500, 50):
        tmp = param.copy()
        tmp['iterations'] = ii
        param_list.append(tmp)
    
    return param_list


def params_cb_tune_depth(param):
    """
    parameter list for tunning depth of catboost model
    """
    param_list = []
    for ii in xrange(1, 11, 1):
        tmp = param.copy()
        tmp['depth'] = ii
        param_list.append(tmp)
    
    return param_list


def params_cb_tune_baggingt(param):
    """
    parameter list for tunning bagging_temperature of catboost model
    """
    param_list = []
    for jj in xrange(0, 11, 1):
        tmp = param.copy()
        tmp['bagging_temperature'] = jj / 10.0
        param_list.append(tmp)
    
    return param_list


def params_cb_tune_depth_baggingt(param):
    """
    parameter list for tunning depth and bagging_temperature of catboost model
    """
    param_list = []
    for ii in xrange(4, 11, 1):
        for jj in xrange(0, 12, 2):
            tmp = param.copy()
            tmp['depth'] = ii
            tmp['bagging_temperature'] = jj / 10.0
            param_list.append(tmp)
    
    return param_list


def params_cb_l2_leaf_reg(param):
    """
    parameter list for tunning l2_leaf_reg of catboost model
    """
    param_list = []
    for ii in xrange(0, 15, 1):
        tmp = param.copy()
        tmp['l2_leaf_reg'] = ii
        param_list.append(tmp)
    
    return param_list


def params_cb_random_strength(param):
    """
    parameter list for tunning random_strength of catboost model
    """
    param_list = []
    for ii in xrange(0, 10, 1):
        tmp = param.copy()
        tmp['random_strength'] = ii
        param_list.append(tmp)
    
    return param_list
    

def params_cb(targetid=None):
    """
    Get the Tuned parameters for xgboost
    """
    param_defaults = []
    
    # for Target 0
    param_defaults.append({'learning_rate': 0.03,
                           'iterations': 1300,
                           'depth':10,
                           'bagging_temperature': 0.8, 
                           'random_strength': 1,
                           'l2_leaf_reg':2})
    
    # for Target 1
    param_defaults.append({'learning_rate': 0.03,
                           'iterations': 1250,
                           'depth': 9,
                           'bagging_temperature': 1, 
                           'random_strength': 1,
                           'l2_leaf_reg':0})
    
    # for Target 2
    param_defaults.append({'learning_rate': 0.03,
                           'iterations': 500,
                           'depth':6,
                           'bagging_temperature': 1, 
                           'random_strength': 1,
                           'l2_leaf_reg':2})
    
    # for Target 3
    param_defaults.append({'learning_rate': 0.03,
                           'iterations': 500,
                           'depth':6,
                           'bagging_temperature': 1, 
                           'random_strength': 1,
                           'l2_leaf_reg':2})
    
    # return
    if targetid is not None:
        return param_defaults[targetid]
    else:
        return param_defaults