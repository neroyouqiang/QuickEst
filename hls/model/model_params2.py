
def params_xgb(targetid=None):
    """
    Get the Tuned parameters for xgboost
    """
    param_defaults = []
    
    # for Target 0 {'subsample': 40, 'colsample_bytree': 94, 'max_depth': 7, 'gamma': 1, 'min_child_weight': 8}
    param_defaults.append({'learning_rate': 0.1,
                           'n_estimators': 100, # 600
                           'max_depth': 5,
                           'min_child_weight': 1,
                           'subsample': 1,
                           'colsample_bytree': 1,
                           'gamma':0 })
    
    # for Target 1 {'subsample': 9, 'colsampweight': 2, 'gamma': 0, 'max_depth': 8}
    param_defaults.append({'learning_rate': 0.1,
                           'n_estimators': 100, # 600
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
    
    
def params_linxgb(targetid=None):
    """
    Get the Tuned parameters for xgboost
    """
    param_defaults = []
    
    # for Target 0 
    param_defaults.append({'learning_rate': 0.1,
                           'n_estimators': 150,
                           'max_depth':5,
                           'min_samples_leaf':1,
                           'subsample': 1, 
                           'gamma':0})
    
    # for Target 1 
    param_defaults.append({'learning_rate': 0.1,
                           'n_estimators': 150,
                           'max_depth':5,
                           'min_samples_leaf':1,
                           'subsample': 1, 
                           'gamma':0})
    
    # for Target 2
    param_defaults.append({'learning_rate': 0.1,
                           'n_estimators': 50,
                           'max_depth':5,
                           'min_samples_leaf':1,
                           'subsample': 1, 
                           'gamma':0})
    
    # for Target 3
    param_defaults.append({'learning_rate': 0.1,
                           'n_estimators': 50,
                           'max_depth':5,
                           'min_samples_leaf':1,
                           'subsample': 1, 
                           'gamma':0})
    
    # return
    if targetid is not None:
        return param_defaults[targetid]
    else:
        return param_defaults
    
    
def params_lasso(targetid=None):
    """
    Get the Tuned parameters for xgboost
    """
    param_defaults = []
    
    # for Target 0
    param_defaults.append({'alpha': 20.0}) # 20.0
    
    # for Target 1
    param_defaults.append({'alpha': 180.0}) # 180.0
    
    # for Target 2
    param_defaults.append({'alpha': 0}) # 1.2
    
    # for Target 3
    param_defaults.append({'alpha': 0}) # 1.0
    
    # return
    if targetid is not None:
        return param_defaults[targetid]
    else:
        return param_defaults
    
    
def params_ann(targetid=None):
    """
    Get the Tuned parameters for xgboost
    """
    param_defaults = []
    
    # for Target 0
    param_defaults.append({'hidden_layer_sizes': (1, 11), 'alpha':20})
    
    # for Target 1
    param_defaults.append({'hidden_layer_sizes': (1, 41), 'alpha':95})
    
    # for Target 2
    param_defaults.append({'hidden_layer_sizes': (30, 30), 'alpha':0})
    
    # for Target 3
    param_defaults.append({'hidden_layer_sizes': (30, 30), 'alpha':0})
    
    # return
    if targetid is not None:
        return param_defaults[targetid]
    else:
        return param_defaults
    
    
    
def params_lasso_outlier(targetid=None):
    """
    Get the Tuned parameters for xgboost
    """
    param_defaults = []
    
    # for Target 0
    param_defaults.append({'alpha': 150.0})
    
    # for Target 1
    param_defaults.append({'alpha': 150.0})
    
    # for Target 2
    param_defaults.append({'alpha': 2.0})
    
    # for Target 3
    param_defaults.append({'alpha': 2.0})
    
    # return
    if targetid is not None:
        return param_defaults[targetid]
    else:
        return param_defaults
    
    
def params_ann_outlier(targetid=None):
    """
    Get the Tuned parameters for xgboost
    """
    param_defaults = []
    
    # for Target 0
    param_defaults.append({'hidden_layer_sizes': (40, 30)})
    
    # for Target 1
    param_defaults.append({'hidden_layer_sizes': (30, 30)})
    
    # for Target 2
    param_defaults.append({'hidden_layer_sizes': (30, 30)})
    
    # for Target 3
    param_defaults.append({'hidden_layer_sizes': (30, 30)})
    
    # return
    if targetid is not None:
        return param_defaults[targetid]
    else:
        return param_defaults