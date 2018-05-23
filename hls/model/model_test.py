import sys
import pickle
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

# functions from original file
import run as originfuncs


def model_testing_previous(model, X, Y, normVal, Y_normalized = True):
    """
    Test the model
    """
    # deal with input
    if len(Y.shape) == 1:
        Y = Y.reshape([-1, 1])
    
    models = [model]
        
    # call the orginal function
    RAEs, RRSEs = originfuncs.model_testing(models, X, Y, [normVal], Y_normalized)
    
    # return
    return RAEs[0], RRSEs[0]


def model_testing(model, X, Y, normVal=None, Y_normalized = True):
    """
    Test the model
    """
    # predict the data
    if model:
        # predict
        # model0 = pickle.load(open("saves/models/mymodel_target0_test", "r"))
        # Y_pre = model.predict(X) + model0.predict(X)
        Y_pre = model.predict(X)
        
        # recover the predictions to the actual scale
        if normVal:
            Y_pre = unnormalization(Y_pre, normVal)
    else:
        Y_pre = X
    
    
    # recover the normalized ground truth if necessary
    if normVal and Y_normalized:
        Y = unnormalization(Y, normVal)
        
    # score the result
    RAE, error = score_REA(Y, Y_pre)
    RRSE, error = score_RRSE(Y, Y_pre)
    
    # return
    return RAE, RRSE, error


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


def unnormalization(data, normval):
    return data * normval['std'] + normval['mean']


def normalization(data, normval):
    return (data + normval['mean']) / normval['std']


def score_REA(Y, Y_pre):
    """
    Score by the true and predicted Y
    """
    Y_mean = np.mean(Y)
    error = Y_pre - Y
    REA = np.mean(np.abs(error)) / (np.mean(np.abs(Y - Y_mean)) + np.finfo(float).eps)
    
    return REA, error


def score_RRSE(Y, Y_pre):
    """
    Score by the true and predicted Y
    """
    Y_mean = np.mean(Y)
    error = Y_pre - Y
    RRSE = LA.norm(error) / LA.norm(Y - Y_mean)
    
    return RRSE, error


def score_MSE(Y, Y_pre):
    """
    Score by the true and predicted Y
    """
    error = Y_pre - Y
    MSE = np.square(error).mean()
    MSE = np.sqrt(MSE)
    
    return MSE, error


def score_Accuracy(Y, Y_pre):
    """
    Score by the true and predicted Y
    """
    results = (Y_pre == Y)
    error = 1 - float(results.sum()) / float(results.size)
    
    return error, results

