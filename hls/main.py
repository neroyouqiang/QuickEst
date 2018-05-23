#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# This file is used to study how the trained model can be used in different devices
import os
import sys
import argparse
import pickle
import numpy as np
# import matplotlib.pyplot as plt

from model.model_data2 import load_train_test_data, load_train_test_data_outlier
from model.model_params2 import params_xgb, params_lasso, params_ann, params_lasso_outlier, params_ann_outlier

from techniques.train_model import TrainTech, FeatureSelectTech, AssembleTech
    

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type = str, default = './data/', 
                    help = 'Directory to the input dataset. ')
parser.add_argument('--predict_area', action = 'store_true',
                    help = 'Run area estimation. ')
parser.add_argument('--classify_timing', action = 'store_true',
                    help = 'Run timing classification. ')
parser.add_argument('--train', action = 'store_true',
                    help = 'Run training for the specified task. ')
parser.add_argument('--store_model_path', type = str, default = './saves/models/techniques.pkl',
                    help = 'Path to store the trained models')
parser.add_argument('--test', action = 'store_true',
                    help = 'Run test for the specified task. Should use with the \'--pretrained_model_path\' option. ')
parser.add_argument('--pretrained_model_path', type = str, default = './saves/models/techniques.pkl',
                    help = 'Path to pretrained models. The models should be dumped in pickle files. ')
parser.add_argument('--feature_select', action = 'store_true',
                    help = 'Use feature selection. If this option is used with \'--train\', the generated model would contain a mask indicating the selected features. If used with \'--test\', then the produced model must provide the mask as well. ')

Data_Type = 'normal' # outlier normal
Target_IDs = [0, 1]
Target_Names = ['LUT', 'FF', 'DSP', 'BRAM']

def get_data(data_dir, test_ids, silence=True):
    """
    Get data by data_type
    """
    if Data_Type == 'outlier':
        X_source, Y_source, X_test, Y_test = load_train_test_data_outlier(split_by='design_give', test_ids=test_ids, drop_ids=[],
                                                                          file_name=data_dir + "/data_outlier.csv")
    else:
        X_source, Y_source, X_test, Y_test = load_train_test_data(split_by='design_give', test_ids=test_ids, drop_ids=[],
                                                                  file_name=data_dir + "/data.csv")
        
    # print data info
    if not silence: print 'Training data number:', X_source.shape[0]
    if not silence: print 'Testing data number:', X_test.shape[0]
    
    # return 
    return X_source, Y_source, X_test, Y_test

  
def get_params(technique, target_id):
    """
    Get the parameters for different technique
    """
    if(type(technique) == TrainTech or type(technique) == FeatureSelectTech):
        if technique.model_name == 'xgb':
            return params_xgb(target_id)
        
        elif technique.model_name == 'lasso':
            if Data_Type == 'outlier':
                return params_lasso_outlier(target_id)
            else:
                return params_lasso(target_id)
            
        elif technique.model_name == 'ann':
            if Data_Type == 'outlier':
                return params_ann_outlier(target_id)
            else:
                return params_ann(target_id)
        
    elif(type(technique) == AssembleTech):
        return {} #
        
    return None 


def show_save_result(result):
    """
    Show and save the result
    """
    
    # print final results
    print "\nTesting RAE results: "
    print result['RAEs'] * 100
    print "Average: "
    print result['RAEs'].mean(axis=0) * 100
                
    print "\nTesting MSE results: "
    print result['MSEs']
    print "Average: "
    print result['MSEs'].mean(axis=0)
                
    print "\nTesting R2 results: "
    print result['R2s']
    print "Average: "
    print result['R2s'].mean(axis=0)
    
    # save file
    """
    if not os.path.exists("saves/tmp/"):
        os.mkdir("saves/tmp/")
        
    np.savetxt('saves/tmp/RAE_list.csv', result['RAEs'], delimiter=',')
    np.savetxt('saves/tmp/MSE_list.csv', result['MSEs'], delimiter=',')
    """


def save_models(techniques, FLAGS, silence=False):
    with open(FLAGS.store_model_path, "wb") as f:
        pickle.dump(techniques, f)
        
    if not silence: print "Models saved to: ", FLAGS.store_model_path
    

def load_models(FLAGS, silence=False):
    if not silence: print "Models loaded from: ", FLAGS.pretrained_model_path
    
    if not os.path.exists(FLAGS.pretrained_model_path):
        sys.exit("Pretrained model file " + FLAGS.pretrained_model_path + " does not exist!")
    with open(FLAGS.pretrained_model_path, "rb") as f:
        techniques = pickle.load(f)  
    return techniques

    
def init_models(val_list, FLAGS):
    """
    Init model
    """
    techniques = []
    for ii in xrange(len(val_list)):
        techniques.append([])
        for target_id in Target_IDs:
            # technique:  xgb, lin, ann, gpr, lasso, ridge, ard
            if FLAGS.feature_select:
                technique = FeatureSelectTech('xgb', sel_method='xgb', sel_threshhold=0.05)
            else:
                technique = TrainTech('xgb')
            
            # technique = FeatureSelectTech('xgb', sel_method='lasso', sel_threshhold=0, lasso_alpha=1.2)
            # technique = AssembleTech(['xgb_selfs', 'lasso'])
            
            # add to list
            techniques[ii].append(technique)
            
    # return
    return techniques
            
            
def train_models(techniques, val_list, FLAGS):
    """
    Train model
    """
    for ii in xrange(len(val_list)):
        
        # load data
        X_source, Y_source, X_test, Y_test = get_data(FLAGS.data_dir, val_list[ii])
        
        # train and test on different targets
        for target_id in Target_IDs:
            # print info
            print "Testing design IDs are", val_list[ii]
            print "Predicting target is", Target_Names[target_id]
            
            # load parameters
            params = get_params(techniques[ii][target_id], target_id)
        
            # train
            techniques[ii][target_id].train(X_source, Y_source[:, target_id], params, random_seed=100)
    
    # return    
    return techniques


def test_models(techniques, val_list, FLAGS):
    """
    Test model
    """
    result = {}
    result['RAEs'] = np.zeros([len(val_list), len(Target_IDs)], dtype=np.float64)
    result['MSEs'] = np.zeros([len(val_list), len(Target_IDs)], dtype=np.float64)
    result['R2s'] = np.zeros([len(val_list), len(Target_IDs)], dtype=np.float64)
    for ii in xrange(len(val_list)):
        # load data
        X_source, Y_source, X_test, Y_test = get_data(FLAGS.data_dir, val_list[ii])
            
        # train and test on different targets
        for target_id in Target_IDs:
            # test
            RAE, MSE, R2 = techniques[ii][target_id].test(X_test, Y_test[:, target_id])
            
            # add to list
            result['RAEs'][ii][target_id] = RAE
            result['MSEs'][ii][target_id] = MSE
            result['R2s'][ii][target_id] = R2
    
    # return
    return result


if __name__ == '__main__':
    # parser
    FLAGS, unparsed = parser.parse_known_args()
    
    # experiment variables
    if Data_Type == 'outlier':
        val_list = [[x for x in xrange(0, 5)], [x for x in xrange(5, 10)], [x for x in xrange(10, 15)], [x for x in xrange(15, 20)]]
    else:
        val_list = [[x for x in xrange(0, 14)], [x for x in xrange(14, 28)], [x for x in xrange(28, 42)], [x for x in xrange(42, 56)]]
    

    # training model
    if FLAGS.train:
        print "\n========== Start training ==========\n"
        
        # init models
        techniques = init_models(val_list, FLAGS)
    
        # train models
        techniques = train_models(techniques, val_list, FLAGS)
        
        # save models
        save_models(techniques, FLAGS)
    
    # testing model
    if FLAGS.test:
        print "\n========== Start testing ==========\n"
        
        # load models
        techniques = load_models(FLAGS)
        
        # test models
        result = test_models(techniques, val_list, FLAGS)
        
        # show and save result
        show_save_result(result)
        
        
    print "\n========== End ==========\n"
    
