#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# This file is used to study how the trained model can be used in different devices
import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

from model.model_data2 import load_train_test_data, load_train_test_data_outlier
from model.model_params2 import params_xgb, params_lasso, params_ann, params_lasso_outlier, params_ann_outlier, params_linxgb
from model.model_test2 import score_REA, score_MSE, score_norm

from techniques.train_model import TrainTech, FeatureSelectTech, BaggingTech, AssembleTech, XGBoostTech
    

parser = argparse.ArgumentParser()
parser.add_argument('--train', action = 'store_true',
                    help = 'Run training for the specified task. ')
parser.add_argument('--store_model_path', type = str, default = './saves/models/techniques.pkl',
                    help = 'Path to store the trained models')
parser.add_argument('--test', action = 'store_true',
                    help = 'Run test for the specified task. Should use with the \'--pretrained_model_path\' option. ')
parser.add_argument('--pretrained_model_path', type = str, default = './saves/models/techniques.pkl',
                    help = 'Path to pretrained models. The models should be dumped in pickle files. ')

Data_Type = 'normal' # outlier normal
Target_Names = ['LUT', 'FF', 'DSP', 'BRAM']

def get_data(test_ids, data_dir='./data/', silence=True):
    """
    Get data by data_type
    """
    if Data_Type == 'outlier':
        X_source, Y_source, X_test, Y_test = load_train_test_data_outlier(split_by='design_give', test_ids=test_ids, drop_ids=[],
                                                                          file_name=data_dir + "/data_outlier.csv", is_normalizeX=True)
    else:
        X_source, Y_source, X_test, Y_test = load_train_test_data(split_by='design_give', test_ids=test_ids, drop_ids=[],
                                                                  file_name=data_dir + "/data.csv", is_normalizeX=True)
        
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
        
        elif technique.model_name == 'linxgb':
            return params_linxgb(target_id)
        
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
        
    return None 


def show_save_result(result, val_list=None):
    """
    Show and save the result
    """
    
    # print final results
    print "\nTesting RAE results: "
    print result['RAEs'] * 100
    print "Average: "
    print result['RAEs'].mean(axis=0) * 100
    print result['RAE_whole'] * 100
                
    print "\nTesting RRSE results: "
    print result['RRSEs'] * 100
    print "Average: "
    print result['RRSEs'].mean(axis=0) * 100
    print result['RRSE_whole'] * 100
    # print (result['MSEs'][:, 2] * result['Test number'][:, 2]).sum() / result['Test number'][:, 2].sum() * 100
                
    print "\nTesting R2 results: "
    print result['R2s']
    print "Average: "
    print result['R2s'].mean(axis=0)
    print result['R2_whole']
    
    # save file
    if not os.path.exists("./saves/results/"):
        os.mkdir("./saves/results/")
        
    # save result
    with open('./saves/results/result.pkl', 'w') as f:
        pickle.dump(result, f, True)
    
    # val_list = np.array(val_list).reshape([-1, 1])
    # np.savetxt('saves/results/val_list.csv', val_list, delimiter=',')
    
    """    
    np.savetxt('saves/results/RAEs.csv', result['RAEs'], delimiter=',')
    np.savetxt('saves/results/MSEs.csv', result['MSEs'], delimiter=',')
    np.savetxt('saves/results/R2s.csv', result['R2s'], delimiter=',')
    if 'RAEs_train' in result: np.savetxt('saves/results/RAEs_train.csv', result['RAEs_train'], delimiter=',')
    if 'MSEs_train' in result: np.savetxt('saves/results/MSEs_train.csv', result['MSEs_train'], delimiter=',')
    if 'R2s_train' in result: np.savetxt('saves/results/R2s_train.csv', result['R2s_train'], delimiter=',')
    """
    
    # result_save = np.hstack([result['RAEs'], result['RAEs_train'], result['MSEs'], result['MSEs_train'], result['R2s'], result['R2s_train']])
    # np.savetxt('saves/results/results.csv', result_save, delimiter=',')
    
    if False:
        plt.figure()
        plt.plot(val_list, result['RAEs_train'][:, 0])
        plt.plot(val_list, result['RAEs'][:, 0])
        plt.legend(['Train', 'Test'])
        plt.show()
        
        plt.figure()
        plt.plot(val_list, result['RAEs_train'][:, 1])
        plt.plot(val_list, result['RAEs'][:, 1])
        plt.legend(['Train', 'Test'])
        plt.show()


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

    
def init_models(val_list, target_ids):
    """
    Init model
    """
    techniques = []
    for ii in xrange(len(val_list)):
        techniques.append([])
        for target_id in xrange(max(target_ids) + 1):
            if target_id in [0, 1]:
                # technique:  xgb, lin, ann, gpr, lasso, ridge, ard
                technique = TrainTech('lasso')
                # technique = FeatureSelectTech('lasso', sel_method='lasso', sel_threshhold=0, lasso_alpha=10.0)
                # technique = FeatureSelectTech('xgb', sel_method='xgb', sel_threshhold=0.03)
                # technique = FeatureSelectTech('xgb', sel_method='lasso', sel_threshhold=0, lasso_alpha=1.2)
                # technique = BaggingTech('lasso', bagging_ratio=0.8, bagging_round=15, def_params_id=target_id)
                # technique = AssembleTech(['xgb_selfs', 'lasso'], def_params_id=target_id)
#                technique = XGBoostTech(def_params_id=target_id, 
#                                        fsel_threshhold=0.03, fsel_round=1, 
#                                        bagging_ratio=0.7, bagging_round=25, bagging_replace=False)
            if target_id in [2, 3]:
                # technique = TrainTech('xgb')
                # technique = FeatureSelectTech('lasso', sel_method='lasso', sel_threshhold=0.4, lasso_alpha=2.0)
                technique = FeatureSelectTech('xgb', sel_method='lasso', sel_threshhold=0.4, lasso_alpha=2.0)
                # technique = AssembleTech(['xgb', 'lasso'])
            
            # add to list
            techniques[ii].append(technique)
            
    # return
    return techniques
            
            
def train_models(techniques, val_list, target_ids, set_params=None, silence=False):
    """
    Train model
    """
    for ii in xrange(len(val_list)):
        # load data
        # test_ids = [x for x in xrange(42, 56)]
        test_ids = val_list[ii]
        X_source, Y_source, X_test, Y_test = get_data(test_ids)
        
        # train and test on different targets
        for target_id in target_ids:
            # print info
            if not silence: print "Testing design IDs are", test_ids
            if not silence: print "Predicting target is", Target_Names[target_id]
            
            # set or load parameters
            if set_params is None:
                params = get_params(techniques[ii][target_id], target_id)
                # params['n_estimators'] = val_list[ii]
            else:
                params = set_params
        
            # train
            techniques[ii][target_id].train(X_source, Y_source[:, target_id], params, random_seed=100)
            
            # save - only the trained model
            techniques[ii][target_id].save_model('./saves/models/model_t%d_g%d.pkl' % (target_id, ii))
    
    # return    
    return techniques


def test_models(techniques, val_list, target_ids):
    """
    Test model
    """
    result = {}
    result['RAEs'] = np.zeros([len(val_list), max(target_ids) + 1], dtype=np.float64)
    result['RRSEs'] = np.zeros([len(val_list), max(target_ids) + 1], dtype=np.float64)
    result['R2s'] = np.zeros([len(val_list), max(target_ids) + 1], dtype=np.float64)
    result['RAEs_train'] = np.zeros([len(val_list), max(target_ids) + 1], dtype=np.float64)
    result['RRSEs_train'] = np.zeros([len(val_list), max(target_ids) + 1], dtype=np.float64)
    result['R2s_train'] = np.zeros([len(val_list), max(target_ids) + 1], dtype=np.float64)
    result['Test number'] = np.zeros([len(val_list), max(target_ids) + 1], dtype=np.float64)
    result['Predict'] = [[], [], [], []]
    result['Truth'] = [[], [], [], []]
    for ii in xrange(len(val_list)):
        # load data
        # test_ids = [x for x in xrange(42, 56)]
        test_ids = val_list[ii]
        X_source, Y_source, X_test, Y_test = get_data(test_ids)
            
        # train and test on different targets
        for target_id in target_ids:
            # on testing set
            RAE, RRSE, R2 = techniques[ii][target_id].test(X_test, Y_test[:, target_id])
            result['RAEs'][ii][target_id] = RAE
            result['RRSEs'][ii][target_id] = RRSE
            result['R2s'][ii][target_id] = R2
            result['Test number'][ii][target_id] = X_test.shape[0]
            
            # on training set
            RAE, RRSE, R2 = techniques[ii][target_id].test(X_source, Y_source[:, target_id])
            result['RAEs_train'][ii][target_id] = RAE
            result['RRSEs_train'][ii][target_id] = RRSE
            result['R2s_train'][ii][target_id] = R2
            
            # np.savetxt('./saves/tmp/Y_true_' + str(ii) + '_' + str(target_id) + '.csv', Y_test[:, target_id])
            # np.savetxt('./saves/tmp/Y_pre_' + str(ii) + '_' + str(target_id) + '.csv', techniques[ii][target_id].predict(X_test))
            
            # predict result
            _predict = techniques[ii][target_id].predict(X_test)
            _truth = Y_test[:, target_id]
            result['Predict'][target_id].extend(_predict)
            result['Truth'][target_id].extend(_truth)
            
    # calculate the result on the whole result
    result['RAE_whole'] = np.zeros([max(target_ids) + 1], dtype=np.float64)
    result['RRSE_whole'] = np.zeros([max(target_ids) + 1], dtype=np.float64)
    result['R2_whole'] = np.zeros([max(target_ids) + 1], dtype=np.float64)
    for target_id in target_ids:
        result['Truth'][target_id] = np.array(result['Truth'][target_id])
        result['Predict'][target_id] = np.array(result['Predict'][target_id])
        
        result['RAE_whole'][target_id], errors = score_REA(result['Truth'][target_id], result['Predict'][target_id])
        result['RRSE_whole'][target_id], errors = score_norm(result['Truth'][target_id], result['Predict'][target_id], norm_ord=2)
        result['R2_whole'][target_id] = metrics.r2_score(result['Truth'][target_id], result['Predict'][target_id])
    
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
        
#        val_list = []
#        for ii in xrange(43): val_list.append([x for x in xrange(ii, ii + 14)])
        
#        val_list = [[x] for x in xrange(56 + 1)]
        
        # by dynamic programming - FF
#        val_list = [[2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19],
#                    [0, 1, 14, 17, 20, 21, 22, 23, 24, 25, 26, 27, 29, 33],
#                    [8, 28, 30, 31, 32, 34, 36, 37, 38, 39, 41, 43, 44, 45],
#                    [11, 35, 40, 42, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]]
        
        # by dynamic programming - LUT
#        val_list = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 17, 18, 19],
#                    [9, 10, 14, 15, 20, 21, 23, 24, 25, 26, 27, 28, 29, 33],
#                    [16, 22, 30, 31, 32, 34, 35, 37, 38, 40, 41, 42, 43, 46],
#                    [13, 36, 39, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]]
        
        # by sorting and selecting - FF
#        val_list = [[3, 5, 7, 19, 22, 23, 29, 31, 35, 37, 38, 39, 42, 53, 54],
#                    [0, 4, 11, 16, 18, 33, 34, 36, 44, 45, 46, 49, 52, 56],
#                    [1, 8, 10, 14, 17, 20, 26, 28, 41, 43, 48, 50, 51, 55],
#                    [2, 6, 9, 12, 13, 15, 21, 24, 25, 27, 30, 32, 40, 47]]
        
        # by sorting and selecting - LUT
#        val_list = [[7, 11, 16, 26, 27, 30, 33, 35, 36, 37, 41, 48, 52, 53, 54], 
#                    [0, 1, 8, 9, 13, 18, 19, 22, 29, 38, 40, 45, 47, 55],
#                    [2, 14, 20, 23, 25, 28, 39, 42, 43, 44, 46, 49, 50, 56],
#                    [3, 4, 5, 6, 10, 12, 15, 17, 21, 24, 31, 32, 34, 51]]
        
        # by random
#        np.random.seed(seed = 101)
#        _tmp = np.random.choice(56, 56, replace=False)
#        val_list = [_tmp[0: 14].tolist(), _tmp[14: 28].tolist(), _tmp[28: 42].tolist(), _tmp[42: 56].tolist()]
        
    
    # val_list = [x for x in xrange(0, 200, 2)]
    target_ids = [0, 1]

    # training model
    if FLAGS.train or True:
        print "\n========== Start training ==========\n"
        
        # init models
        techniques = init_models(val_list, target_ids)
    
        # train models
        techniques = train_models(techniques, val_list, target_ids)
        
        # save models
        # save_models(techniques, FLAGS)
    
    # testing model
    if FLAGS.test or True:
        print "\n========== Start testing ==========\n"
        
        # load models
        # techniques = load_models(FLAGS)
        
        # test models
        result = test_models(techniques, val_list, target_ids)
        
        # show and save result
        show_save_result(result, val_list)
        
        
    print "\n========== End ==========\n"
    
