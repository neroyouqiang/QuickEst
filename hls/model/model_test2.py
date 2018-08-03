import sys
import pickle
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA


def model_testing_RAE(model, X, Y):
    """
    Test the model
    """
    # predict the data
    if model:
        # predict
        Y_pre = model.predict(X)
    else:
        Y_pre = X
        
    # score the result
    RAE, error = score_REA(Y, Y_pre)
    
    # return
    return RAE, error


def model_testing_MSE(model, X, Y):
    """
    Test the model
    """
    # predict the data
    if model:
        # predict
        Y_pre = model.predict(X)
    else:
        Y_pre = X
        
    # score the result
    MSE, error = score_MSE(Y, Y_pre)
    
    # return
    return MSE, error


def model_testing_timing(model, X, Y):
    """
    Test the model
    """
    # print X.shape
    Y_pre = model.predict(X)
        
    # score the result
    error, results = score_Accuracy(Y, Y_pre)
    
    # return
    return error, results


def score_REA(Y, Y_pre):
    """
    Score by the true and predicted Y
    """
    Y_pre = np.array(Y_pre)
    Y = np.array(Y)
    
    Y_mean = np.mean(Y)
    error = Y_pre - Y
    RAE = np.mean(np.abs(error)) / (np.mean(np.abs(Y - Y_mean)) + np.finfo(float).eps)
    
    # return np.mean(Y), error
    return RAE, error


def score_RRSE(Y, Y_pre):
    """
    Score by the true and predicted Y
    """
    Y_pre = np.array(Y_pre)
    Y = np.array(Y)
    
    Y_mean = np.mean(Y)
    error = Y_pre - Y
    RRSE = LA.norm(error) / LA.norm(Y - Y_mean)
    
    return RRSE, error


def score_MSE(Y, Y_pre):
    """
    Score by the true and predicted Y
    """
    Y_pre = np.array(Y_pre)
    Y = np.array(Y)
    
    error = Y_pre - Y
    MSE = np.square(error).mean()
    MSE = np.sqrt(MSE)
    
    # Y_mean = np.mean(Y)
    # return np.std(Y), error
    return MSE, error


def score_Accuracy(Y, Y_pre):
    """
    Score by the true and predicted Y
    """
    Y_pre = np.array(Y_pre)
    Y = np.array(Y)
    
    results = (Y_pre == Y)
    error = 1 - float(results.sum()) / float(results.size)
    
    return error, results


def score_norm(Y, Y_pre, norm_ord=2):
    """
    Score by the true and predicted Y
    """
    Y_pre = np.array(Y_pre)
    Y = np.array(Y)
    
    Y_mean = np.mean(Y)
    error = Y_pre - Y
    
    RAE = LA.norm(error, ord=norm_ord) / (LA.norm(Y - Y_mean, ord=norm_ord) + np.finfo(float).eps)
    
    # return np.mean(Y), error
    return RAE, error

# -*- coding: utf-8 -*-

