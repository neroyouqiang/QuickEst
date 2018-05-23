# This file is used to load data and compatible with the functions written of Zhou
import sys
import numpy as np
import matplotlib.pyplot as plt

# functions from original file
import run as originfuncs


def data_load(Y_start_col=2, Y_end_col=6):
    """
    Load orginal data.
    """
    # read input path
    """
    if len(sys.argv) < 2:
        # sys.exit("Please provide input data path!")
        input_path = "./data_preprocessed/"
    else:
        input_path = sys.argv[1]
    """
    input_path = "./data_preprocessed/"

    # load data from pickle file
    all_data, dev_data, names = originfuncs.load_data(input_path, Y_start_col, Y_end_col, silence=True)
    
    # return
    return all_data, dev_data, names

"""
def data_prepare_normvals(target_id=None, select_feature=True):
    # load data
    data_all, data_dev, data_names = data_load()
    
    norm_X = data_all['normVal']['features']
    norm_Y = data_all['normVal']['training_Y']
    
    # select y
    if target_id:
        norm_Y = norm_Y[:, target_id]
    
    # select features
    if select_feature:
        # load inex of selected features
        selected_features = np.loadtxt("saves/selected_features_index.csv", delimiter=",", dtype=int)
        
        # cut out data
        norm_X = norm_X[selected_features]
    
    return norm_X, norm_Y
""" 

def load_data_all(select_feature=True, is_normalizeX=True):
    """
    Load all the X and Y
    """
    # load data
    data_all, data_dev, data_names = data_load()
    
    X_train = data_all['training_X']
    Y_train = data_all['training_Y']
    X_test = data_all['test_X']
    Y_test = data_all['test_Y']
    norm_X = data_all['normVal']['features']
    norm_Y = data_all['normVal']['training_Y']
    
    # renormalize y
    # for ii in xrange(Y_train.shape[1]):
    #     Y_train[:, ii] = Y_train[:, ii] * norm_Y[ii]['std'] + norm_Y[ii]['mean']
        
    # renormalize x
    if is_normalizeX:
        for ii in xrange(X_train.shape[1]):
            X_train[:, ii] = X_train[:, ii] * norm_X[ii]['std'] + norm_X[ii]['mean']
            X_test[:, ii] = X_test[:, ii] * norm_X[ii]['std'] + norm_X[ii]['mean']
            
    # select features
    if select_feature:
        # load inex of selected features
        selected_features = np.loadtxt("saves/selected_features_index.csv", delimiter=",", dtype=int)
        
        # cut out data
        X_train = X_train[:, selected_features]
        X_test = X_test[:, selected_features]
        data_names['features_name'] = data_names['features_name'][selected_features]
    
    # all X and Y
    X = np.vstack((X_train, X_test))
    Y = np.vstack((Y_train, Y_test))
    
    # return
    return X, Y, norm_Y, data_names


def load_data_devs(select_feature=True, is_normalizeX=True, is_normalizeY=False):
    """
    Load all X and Y divided by devices
    """
    # load data
    data_all, data_dev, data_names = data_load()
    
    # normalizing data
    norm_X = data_all['normVal']['features']
    norm_Y = data_all['normVal']['training_Y']
    
    # data
    X_devs = []
    Y_devs = []
    for ii in xrange(len(data_dev)):
        X_train = data_dev[ii]['training_X']
        Y_train = data_dev[ii]['training_Y']
        X_test = data_dev[ii]['test_X']
        Y_test = data_dev[ii]['test_Y']
            
        # normalize x
        if not is_normalizeX:
            for ii in xrange(X_train.shape[1]):
                X_train[:, ii] = X_train[:, ii] * norm_X[ii]['std'] + norm_X[ii]['mean']
                X_test[:, ii] = X_test[:, ii] * norm_X[ii]['std'] + norm_X[ii]['mean']
                
        # normalize y
        if is_normalizeY:
            for ii in xrange(Y_train.shape[1]):
                Y_train[:, ii] = (Y_train[:, ii] - norm_Y[ii]['mean']) / norm_Y[ii]['std']
                Y_test[:, ii] = (Y_test[:, ii] - norm_Y[ii]['mean']) / norm_Y[ii]['std']
                
        # select features
        if select_feature:
            # load inex of selected features
            selected_features = np.loadtxt("saves/selected_features_index.csv", delimiter=",", dtype=int)
            
            # cut out data
            X_train = X_train[:, selected_features]
            X_test = X_test[:, selected_features]
        
        # all X and Y
        X = np.vstack((X_train, X_test))
        Y = np.vstack((Y_train, Y_test))
        
        # add to list
        X_devs.append(X.copy())
        Y_devs.append(Y.copy())
     
    # select features
    if select_feature:   
        data_names['features_name'] = data_names['features_name'][selected_features]
    
    # return
    return X_devs, Y_devs, norm_Y, data_names
    

def load_train_test_data(split_by='dev', test_device=3, test_ratio=0.2, 
                         select_feature=True, is_normalizeX=True):
    """
    Load test and train data by different splitting methods.
    """
    if split_by == "dev":
        # load data
        X_devs, Y_devs, norm_Y, data_names = load_data_devs(select_feature=select_feature,
                                                            is_normalizeX=is_normalizeX)
        
        # testing data
        X_test = X_devs[test_device]
        Y_test = Y_devs[test_device]
        
        # training data
        X_train = None
        Y_train = None
        
        # vstack training data
        for ii in xrange(4):
            if ii != test_device:
                if X_train is None:
                    X_train = X_devs[ii]
                else:
                    X_train = np.vstack((X_train, X_devs[ii]))
                    
                if Y_train is None:
                    Y_train = Y_devs[ii]
                else:
                    Y_train = np.vstack((Y_train, Y_devs[ii]))
                    
    elif split_by == 'ratio':
        # load data
        X_all, Y_all, norm_Y, data_names = load_data_all(select_feature=select_feature,
                                                         is_normalizeX=is_normalizeX)
        
        # shuffle data
        shuffled_index = np.random.permutation(X_all.shape[0])
        X_all = X_all[shuffled_index]
        Y_all = Y_all[shuffled_index]
        
        # training data
        X_train = X_all[0: int(X_all.shape[0] * (1 - test_ratio)) + 1]
        Y_train = Y_all[0: int(Y_all.shape[0] * (1 - test_ratio)) + 1]
        
        # testing data
        X_test = X_all[int(X_all.shape[0] * (1 - test_ratio)) + 1 ::]
        Y_test = Y_all[int(Y_all.shape[0] * (1 - test_ratio)) + 1 ::]
         
    elif split_by == 'ratio_device':
        # load data
        X_devs, Y_devs, norm_Y, data_names = load_data_devs(select_feature=select_feature,
                                                            is_normalizeX=is_normalizeX)

        # testing data
        X_test = None
        Y_test = None
        
        # training data
        X_train = None
        Y_train = None
        
        for ii in xrange(4):
            _x_train = X_devs[ii][0: int(np.round(X_devs[ii].shape[0] * (1 - test_ratio)))]
            _y_train = Y_devs[ii][0: int(np.round(Y_devs[ii].shape[0] * (1 - test_ratio)))]
            
            _x_test = X_devs[ii][int(np.round(X_devs[ii].shape[0] * (1 - test_ratio)))::]
            _y_test = Y_devs[ii][int(np.round(Y_devs[ii].shape[0] * (1 - test_ratio)))::]
            
            if X_train is None:
                X_train = _x_train
                Y_train = _y_train
                X_test = _x_test
                Y_test = _y_test
            else:
                X_train = np.vstack((X_train, _x_train))
                Y_train = np.vstack((Y_train, _y_train))
                X_test = np.vstack((X_test, _x_test))
                Y_test = np.vstack((Y_test, _y_test))
                
    # return data
    return X_train, Y_train, X_test, Y_test, norm_Y, data_names 

    
def data_prepare(target_id=None, select_feature=True, is_normalizeX=True):
    """
    Prepare data for training, tuning, testing model
    
    Parameters
    ----------
    target_id : int. {None, 0, 1, 2, 3}
        ID of the target. If none, select all targets.
        
    select_feature : bool
        If true, select the important features
        
    Returns
    -------
    X, Y, X_test, Y_test, X_test_dev, Y_test_dev, norm_Y, data_names
    """
    # load data
    data_all, data_dev, data_names = data_load()
    
    X_train = data_all['training_X']            # normalized
    Y_train = data_all['norm_training_Y']       # normalized
    X_test = data_all['test_X']                     # normalized
    Y_test = data_all['test_Y']                     # NOT NORMALIZED!!!
    norm_X = data_all['normVal']['features']
    norm_Y = data_all['normVal']['training_Y']
    
    # prepare training data - all data
    if target_id is not None:
        Y_train = Y_train[:, target_id]
        norm_Y = norm_Y[target_id]
        
    if not is_normalizeX:
        for ii in xrange(X_train.shape[1]):
            X_train[:, ii] = X_train[:, ii] * norm_X[ii]['std'] + norm_X[ii]['mean']
    
    # prepare testing data - all data
    if target_id is not None:
        Y_test = Y_test[:, target_id]
    
    if not is_normalizeX:
        for ii in xrange(X_train.shape[1]):
            X_test[:, ii] = X_test[:, ii] * norm_X[ii]['std'] + norm_X[ii]['mean']
    
    # prepare testing data - for device
    X_test_devs = []
    Y_test_devs = []
    for ii in xrange(len(data_dev)):
        X_test_dev = data_dev[ii]['test_X']
        Y_test_dev = data_dev[ii]['test_Y']
        
        if target_id is not None:
            Y_test_dev = Y_test_dev[:, target_id]
        
        if not is_normalizeX:
            for ii in xrange(X_train.shape[1]):
                X_test_dev[:, ii] = X_test_dev[:, ii] * norm_X[ii]['std'] + norm_X[ii]['mean']
                
        X_test_devs.append(X_test_dev.copy())
        Y_test_devs.append(Y_test_dev.copy())
    
    # select features
    if select_feature:
        # load inex of selected features
        selected_features = np.loadtxt("saves/selected_features_index.csv", delimiter=",", dtype=int)
        
        # cut out data
        X_train = X_train[:, selected_features]
        X_test = X_test[:, selected_features]
        data_names['features_name'] = data_names['features_name'][selected_features]
        for ii in xrange(len(data_dev)):
            X_test_devs[ii] = X_test_devs[ii][:, selected_features]
            
        # cut out data2
        """
        selected_features2 = [1, 2, 3, 4, 5]
        X_train = X_train[:, selected_features2]
        X_test = X_test[:, selected_features2]
        for ii in xrange(len(data_dev)):
            X_test_devs[ii] = X_test_devs[ii][:, selected_features2]
        """
        
    # return
    return X_train, Y_train, X_test, Y_test, X_test_devs, Y_test_devs, norm_Y, data_names


def data_prepare_timing(select_feature=True, is_normalizeX=True):
    """
    Prepare data for training, tuning, testing model. Timing prediction.
    """
    # load data
    data_all, data_dev, data_names = data_load(Y_start_col=0, Y_end_col=1)
    
    norm_X = data_all['normVal']['features']
    # norm_Y = data_all['normVal']['training_Y']
    
    # prepare training data
    CP_Implement = data_all['training_Y'][:, 0]
    CP_Target = data_all['training_X'][:, 12] * norm_X[12]['std'] + norm_X[12]['mean']
    
    X_train = data_all['training_X']
    Y_train = CP_Target > CP_Implement
    
    # prepare testing data
    CP_Implement = data_all['test_Y'][:, 0]
    CP_Target = data_all['test_X'][:, 12] * norm_X[12]['std'] + norm_X[12]['mean']
    
    X_test = data_all['test_X']
    Y_test = CP_Target > CP_Implement
    
    # prepare testing data - for device
    X_test_devs = []
    Y_test_devs = []
    for ii in xrange(len(data_dev)):
        CP_Implement = data_dev[ii]['test_Y'][:, 0]
        CP_Target = data_dev[ii]['test_X'][:, 12] * norm_X[12]['std'] + norm_X[12]['mean']
        
        X_test_dev = data_dev[ii]['test_X']
        Y_test_dev = CP_Target > CP_Implement
                
        X_test_devs.append(X_test_dev.copy())
        Y_test_devs.append(Y_test_dev.copy())
    
    # unnormalize X
    if not is_normalizeX:
        for ii in xrange(X_train.shape[1]):
            X_train[:, ii] = X_train[:, ii] * norm_X[ii]['std'] + norm_X[ii]['mean']
        
        for ii in xrange(X_train.shape[1]):
            X_test[:, ii] = X_test[:, ii] * norm_X[ii]['std'] + norm_X[ii]['mean']
            
        for ii in xrange(len(data_dev)):
            for ii in xrange(X_train.shape[1]):
                X_test_dev[:, ii] = X_test_dev[:, ii] * norm_X[ii]['std'] + norm_X[ii]['mean']
            
    # select features
    if select_feature:
        # load inex of selected features
        selected_features = np.loadtxt("saves/selected_features_index_timing.csv", delimiter=",", dtype=int)
        
        # cut out data
        X_train = X_train[:, selected_features]
        X_test = X_test[:, selected_features]
        data_names['features_name'] = data_names['features_name'][selected_features]
        for ii in xrange(len(data_dev)):
            X_test_devs[ii] = X_test_devs[ii][:, selected_features]
        
    # return
    return X_train, Y_train, X_test, Y_test, X_test_devs, Y_test_devs, data_names


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test, X_test_devs, Y_test_devs, data_names = data_prepare_timing(select_feature=False)