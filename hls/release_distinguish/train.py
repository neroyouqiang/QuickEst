# -*- coding: utf-8 -*-
import os
import pickle
import sys
import argparse

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type = str, default = './data/data_train.pkl', 
                    help = 'Directory to the training dataset. ')
parser.add_argument('--save_model_dir', type = str, default = './saves/train/models.pkl', 
                    help = 'Directory to save the trained model. Input folder or file name.')
# parser.add_argument('--feature_select', action = 'store_true',
#                     help = 'Use feature selection. ')


def load_data(file_name, silence=False):
    if not silence: print ''
    if not silence: print "Load data from: ", file_name
    
    # check file exist
    if not os.path.exists(file_name):
        sys.exit("Data file " + file_name + " does not exist!")
    
    # load data
    with open(file_name, "rb") as f:
        data = pickle.load(f)
        
    return data[0], data[1]


def save_models(file_save, models, silence=False):
    # input file name
    file_dir, file_name = os.path.split(file_save)
    if file_dir == '': file_dir = "./saves/train/"
    if file_name == '': file_name = 'models.pkl'
    
    # create folder
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    
    # create file
    with open(os.path.join(file_dir, file_name), "wb") as f:
        pickle.dump(models, f)
        
    if not silence: print ''
    if not silence: print "Save models to: ", os.path.join(file_dir, file_name)
    
    
def train_model(X, Y, sel_features=None, silence=False):
    if not silence: print ''
    if not silence: print 'Training model XGBClassifier begins ...'
    
    # select features
    if sel_features is not None:
        X = X[sel_features].copy()
    else:
        X = X.copy()
    
    # fix the random seed
    np.random.seed(seed = 100)
    
    # init
    models = []
    models.append(xgb.XGBClassifier())
    # models.append(LogisticRegression(class_weight={0:1, 1:1}))
    # models.append(svm.SVC())
    # models.append(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30, 30)))
    
    # fit
    models[0].fit(X, Y)
    # models[1].fit(X, Y)
    # models[2].fit(X, Y)
    
    if not silence: print 'Training model XGBClassifier ends.'
    
    # return 
    return {'models':models, 'sel_features':sel_features}
        

def feature_select(X, Y, silence=False):
    if not silence: print ''
    if not silence: print 'Selecting features begins ...'
    
    X = X.copy()
    
    for ii in xrange(40):
        # init
        model = LogisticRegression()
    
        # fit
        model.fit(X, Y)
        
        # select features
        sel_features = X.columns[np.abs(model.coef_[0]) > np.abs(model.coef_[0]).min()]
        sel_features = list(sel_features)
        
        # refresh X
        X = X[sel_features]
        
        # print sel_features
        # print len(sel_features)
    
    if not silence: print 'Selected features are', sel_features
    if not silence: print 'Selecting features ends.'
    
    # return 
    return sel_features
    

if __name__ == '__main__':
    # parser
    FLAGS, unparsed = parser.parse_known_args()
    
    # print info
    print "\n========== Start training models =========="
    
    # load training data
    X, Y = load_data(FLAGS.data_dir)
    
    # select features
    sel_features = feature_select(X, Y['IsOutlier'])
    
    # train
    models = train_model(X, Y['IsOutlier'], sel_features=sel_features)
    
    # save models
    save_models(FLAGS.save_model_dir, models)

    print "\n========== End ==========\n"
