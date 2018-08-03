import pickle

# boosting methods
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import linxgboost.linxgb as linxgb

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ARDRegression
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def model_training(X, Y, model_name, hyperparams={}, silence=False):
    
    if not silence: print "Train model", model_name, "with parameters", hyperparams
    
    # train models
    model = None
    if model_name == 'lin' or model_name == 'linear':
        model = LinearRegression()
        
    if model_name == 'lasso' or model_name == 'lassolin':
        model = Lasso(alpha=hyperparams['alpha'])
        
    if model_name == 'ridge' or model_name == 'ridgelin':
        model = Ridge(alpha=10.0)
        
    if model_name == 'ard' or model_name == 'ardlin':
        model = ARDRegression()
    
    if model_name == 'ann' or model_name == 'artificialneuralnetwork': 
        model = MLPRegressor(solver='lbfgs', hidden_layer_sizes=hyperparams['hidden_layer_sizes'], random_state=0)
        
    elif model_name == 'xgb' or model_name == 'xgboost':
        model = xgb.XGBRegressor(learning_rate=hyperparams['learning_rate'],
                                 n_estimators=hyperparams['n_estimators'],
                                 max_depth=hyperparams['max_depth'],
                                 min_child_weight=hyperparams['min_child_weight'],
                                 subsample=hyperparams['subsample'],
                                 colsample_bytree=hyperparams['colsample_bytree'],
                                 gamma=hyperparams['gamma'])
        
    elif model_name == 'cb' or model_name == 'catboost':
        model = cb.CatBoostRegressor(logging_level='Silent', 
                                     learning_rate=hyperparams['learning_rate'], 
                                     iterations=hyperparams['iterations'], 
                                     depth=hyperparams['depth'],
                                     bagging_temperature=hyperparams['bagging_temperature'],
                                     random_strength=hyperparams['random_strength'], 
                                     l2_leaf_reg=hyperparams['l2_leaf_reg'])
        
    elif model_name == 'linxgb' or model_name == 'linxgboost':
        model = linxgb.linxgb(learning_rate=hyperparams['learning_rate'],
                               n_estimators=hyperparams['n_estimators'],
                               max_depth=hyperparams['max_depth'],
                               subsample=hyperparams['subsample'],
                               min_samples_leaf=hyperparams['min_samples_leaf'],
                               gamma=hyperparams['gamma'])
                               # colsample_bytree=hyperparams['colsample_bytree'],
    elif model_name == 'gpr':
        _kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=_kernel, alpha=1e-3)
        
    model.fit(X, Y)
    
    # return the trained model
    return model


def model_training_timing(X, Y, model_name, hyperparams={}, silence=False):
    
    if not silence: print "Train model", model_name, "with parameters", hyperparams
    
    # train models
    model = None
    if model_name == 'xgb' or model_name == 'xgboost':
        model = xgb.XGBClassifier(learning_rate=hyperparams['learning_rate'],
                                 n_estimators=hyperparams['n_estimators'],
                                 max_depth=hyperparams['max_depth'],
                                 min_child_weight=hyperparams['min_child_weight'],
                                 subsample=hyperparams['subsample'],
                                 colsample_bytree=hyperparams['colsample_bytree'],
                                 gamma=hyperparams['gamma'])
        
    model.fit(X, Y)
    
    # return the trained model
    return model
