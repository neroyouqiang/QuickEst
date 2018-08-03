# -*- coding: utf-8 -*-
import os
import pickle
import sys
import argparse
import pandas as pd

import numpy as np
from sklearn import metrics


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type = str, default = './data/data_test.pkl', 
                    help = 'Directory to the testing dataset. ')
parser.add_argument('--model_dir', type = str, default = './saves/train/models.pkl', 
                    help = 'Directory to the pre-trained models. ')
parser.add_argument('--save_result_dir', type = str, default = './saves/test/results.csv', 
                    help = 'Directory to save the result. Input folder or file name.')


def score_REA(Y, Y_pre):
    """
    Score by the true and predicted Y
    """
    Y_pre = np.array(Y_pre)
    Y = np.array(Y)
    
    Y_mean = np.mean(Y)
    error = Y_pre - Y
    REA = np.mean(np.abs(error)) / (np.mean(np.abs(Y - Y_mean)) + np.finfo(float).eps)
    
    return REA


def load_data(file_name, silence=False):
    if not silence: print ''
    if not silence: print "Load data from: ", file_name
    
    if not os.path.exists(file_name):
        sys.exit("Data file " + file_name + " does not exist!")
        
    with open(file_name, "rb") as f:
        data = pickle.load(f)  
        
    return data[0], data[1]


def load_model(file_name, silence=False):
    if not silence: print ''
    if not silence: print "Load model from: ", file_name
    
    if not os.path.exists(file_name):
        sys.exit("Model file " + file_name + " does not exist!")
        
    with open(file_name, "rb") as f:
        models = pickle.load(f)  
        
    return models


def save_results(file_save, results, silence=False):
    if not os.path.exists("./saves/test/"):
        os.mkdir("./saves/test/")
        
    # input file name
    file_dir, file_name = os.path.split(file_save)
    if file_dir == '': file_dir = "./saves/train/"
    if file_name == '': file_name = 'models.pkl'
    
    # create folder
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
        
    # create file
    np.savetxt(os.path.join(file_dir, file_name), results, delimiter=',', header='Is Outlier, Probability', comments='')
        
    if not silence: print ''
    if not silence: print "Save results to: ", os.path.join(file_dir, file_name)


def test_model(models, X, Y, silence=False):
    
    # model predict
    Y_pre, Y_pre_proba = models_predict(X, models)
        
    # metric - accuracy
    # errors = Y == Y_pre
    # accuracy = float(errors.sum()) / len(errors)
    
    # metric - AUC
    if len(Y_pre_proba.shape) == 1:
        auc = metrics.roc_auc_score(Y, Y_pre_proba[:])
    else:
        auc = metrics.roc_auc_score(Y, Y_pre_proba[:, 1])
        
    # feature importance
    features = pd.DataFrame()
    features['Name'] = X[models['sel_features']].columns
    try:
        b = models['models'][0].get_booster()
        feature_weights = [b.get_score(importance_type='weight').get(f1, 0.) for f1 in b.feature_names]
        feature_weights = np.array(feature_weights, dtype=np.float32)
        features['Importance'] = feature_weights
    except Exception:
        features['Importance'] = models['models'][0].coef_
         
    features = features.sort_values('Importance')
    
    if not silence: print ""
    if not silence: print "AUC of the results are: ", auc
    if not silence: print "The important features are: ", features
    
    # return 
    return np.concatenate((Y_pre.reshape(-1, 1), Y_pre_proba[:, 1:2]), axis=1)


def models_predict(X, models):
    sel_features = models['sel_features']
    models = models['models']
    
    if sel_features is not None:
        X = X[sel_features].copy()
    else:
        X = X.copy()
        
    # predict
    Y_pres = []
    Y_pres_proba = []
    for ii in xrange(len(models)):
        Y_pre = models[ii].predict(X)
        Y_pres.append(Y_pre)
        try:
            Y_pres_proba.append(models[ii].predict_proba(X))
        except Exception:
            Y_pres_proba.append(Y_pre)
            print "predict_proba is called wrong"
        
    # print Y_pres
    Y_pre = np.array(Y_pres).mean(axis=0) >= 0.5
    Y_pre_proba = np.array(Y_pres_proba).mean(axis=0)
    
    # return
    return Y_pre, Y_pre_proba


if __name__ == '__main__':
    # parser
    FLAGS, unparsed = parser.parse_known_args()
    
    # print info
    print "\n========== Start testing models =========="
    
    # load testing data
    X, Y = load_data(FLAGS.data_dir)
    
    # load model
    models = load_model(FLAGS.model_dir)
    
    # train models
    results = test_model(models, X, Y['IsOutlier'])
    
    # save results
    save_results(FLAGS.save_result_dir, results)
    
    print "\n========== End ==========\n"

