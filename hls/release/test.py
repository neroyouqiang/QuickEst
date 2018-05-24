# -*- coding: utf-8 -*-
import os
import pickle
import sys
import argparse

import numpy as np
from sklearn import metrics


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type = str, default = './data/data_test.pkl', 
                    help = 'Directory to the input dataset. ')
parser.add_argument('--model_dir', type = str, default = './saves/train/models.pkl', 
                    help = 'Directory to the input dataset. ')


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
    if not silence: print "Load data from: ", file_name
    
    if not os.path.exists(file_name):
        sys.exit("Data file " + file_name + " does not exist!")
        
    with open(file_name, "rb") as f:
        data = pickle.load(f)  
        
    return data[0], data[1]


def load_model(file_name, silence=False):
    if not silence: print "Load model from: ", file_name
    
    if not os.path.exists(file_name):
        sys.exit("Model file " + file_name + " does not exist!")
        
    with open(file_name, "rb") as f:
        models = pickle.load(f)  
        
    return models


def save_results(results, silence=False):
    if not os.path.exists("./saves/test/"):
        os.mkdir("./saves/test/")
        
    np.savetxt('./saves/test/results.csv', results, delimiter=',')
        
    if not silence: print "Results are saved to: ", "./saves/test/results.csv"


def test_models(models, X, Y, silence=False):
    # test model
    results = np.zeros([X.shape[0], len(models)], dtype=np.float)
    scores = np.zeros([len(models)], dtype=np.float)
    for ii in xrange(len(models)):
        # load models
        model_xgb = models[ii][0]
        model_lasso = models[ii][1]
        features = models[ii][2]
        
        # predict
        results0 = model_xgb.predict(X[:, features])
        retults1 = model_lasso.predict(X)
        
        results[:, ii] = (results0 + retults1) / 2.0
        # scores[ii] = metrics.r2_score(Y[:, ii], results[:, ii])
        scores[ii] = score_REA(Y[:, ii], results[:, ii])
    
    # print scores
    if not silence: print "Score of the results are: ", scores
    
    # return 
    return results
    

if __name__ == '__main__':
    # fix the random seed
    np.random.seed(seed = 6)
    
    # print info
    print "\n========== Start testing models ==========\n"
    
    # parser
    FLAGS, unparsed = parser.parse_known_args()
    
    # load testing data
    X, Y = load_data(FLAGS.data_dir)
    
    # load model
    models = load_model(FLAGS.model_dir)
    
    # train models
    results = test_models(models, X, Y)
    
    # save results
    save_results(results)
    
    print "\n========== End ==========\n"
