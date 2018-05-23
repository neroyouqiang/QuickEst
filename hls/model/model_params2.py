
def params_xgb(targetid=None):
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
    
    
def params_lasso(targetid=None):
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
    
    
def params_ann(targetid=None):
    """
    Get the Tuned parameters for xgboost
    """
    param_defaults = []
    
    # for Target 0
    param_defaults.append({'hidden_layer_sizes': (30, 30)})
    
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