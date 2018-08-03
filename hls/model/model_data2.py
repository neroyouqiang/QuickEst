# -*- coding: utf-8 -*-
import os
import random
import pandas as pd
import numpy as np
import pickle

#features_lasso_lut = ['FF', 'LUT', 'DSP48E_avail', 'FIFO_LUT', 'MEM_TOTAL_TOTAL', 
#                      'MUX_TOTALBITS_TOTAL', 'sub_op_lut', 'add_sum', 'ext_ff',
#                      'dadd_ff', 'dadd_dsp']

def load_data(is_normalizeX=True, is_normalizeY=False, 
              is_select_features=True, select_features=None, is_shuffle_design=True,
              file_name='./data/data.csv'):
    
    df = pd.read_csv(file_name, sep=',')
    df = df.drop(['Main_Report_Path', 'Design_Path'], axis=1)
    
    df_targets = df[['LUT_impl', 'FF_impl', 'DSP_impl', 'BRAM_impl', 'CP_impl']].copy() # 'SLICE_impl', 'SRL_impl', 
    df_features = df[df.columns[2: -14]].copy()
    
    # change column name
    df_features = df_features.rename({'BRAM_18K':'BRAM', 'DSP48E':'DSP'}, axis='columns')
        
    try:
        df_features.rename({'BRAM_18K':'BRAM'}, axis='columns')
    except Exception:
        pass
        
    try:
        df_features.rename({'DSP48E':'DSP'}, axis='columns')
    except Exception:
        pass
    
    # normalization
    if is_normalizeX:
        # df_features = df_features.apply(lambda x: x ** 0.01)
        df_features = (df_features - df_features.mean()) / (df_features.std() + 1e-6)
        
    if is_normalizeY:
        df_targets = (df_targets - df_targets.mean()) / (df_targets.std() + 1e-6)
        
    # select feature
    if is_select_features:
        # select features
        if select_features is None:
            """
            selected_features = np.loadtxt("saves/selected_features_index.csv", delimiter=",", dtype=int)
            df_features = df_features.iloc[:, selected_features]
            """
           
            # drop features
            drop = np.loadtxt('./saves/drop_features_1.csv', dtype=np.str)
            df_features = df_features.drop(drop, axis=1)
        else:
            # select
            df_features = df_features[select_features]
    
    # shuffle the design id
    map_design = pd.DataFrame()
    map_design['Original'] = pd.Series(df['Design_Index'].unique()).sort_values()
    map_design['Original'] = map_design['Original']
    map_design = map_design.set_index('Original')
    
    if is_shuffle_design:
        # map_design_minindex = map_design.index.min()
        # map_design.index = map_design.index - map_design_minindex
        
        # map_design['New'] = map_design.index
        # random.Random(7).shuffle(map_design['New'])
        
        # map_design.index = map_design.index + map_design_minindex
    
        # map_design.to_csv('./saves/map_design_id.csv')
        
        map_design_input = np.loadtxt('./model/saves/map_design_id_v5.csv', delimiter=',', dtype=int)
        map_design = pd.Series(data=map_design_input[:, 1], index=map_design_input[:, 0])
    else:
        map_design['New'] = map_design.index
        
    # df_features = df_features[features_lasso_lut]
    
    # add index
    df_features['Design_Index'] = df['Design_Index'].apply(lambda x:map_design.loc[x]).copy()
    df_features['Device_Index'] = df['Target_Dev_Family'].copy()
    
    df_targets['Design_Index'] = df['Design_Index'].apply(lambda x:map_design.loc[x]).copy()
    df_targets['Device_Index'] = df['Target_Dev_Family'].copy()
    
    
#    with open('../saves/data/df_features.pkl', 'w') as f:
#        pickle.dump(df_features, f, True)
#    with open('../saves/data/df_targets.pkl', 'w') as f:
#        pickle.dump(df_targets, f, True)
    
    # return
    return df_features, df_targets


def load_data_outlier(is_normalizeX=True, is_normalizeY=False, 
                      is_select_features=True, select_features=None, file_name='./data/data_outlier.csv'):
    
    return load_data(is_normalizeX=is_normalizeX, 
                     is_normalizeY=is_normalizeY, 
                     is_select_features=is_select_features,
                     select_features=select_features,
                     is_shuffle_design=False, 
                     file_name=file_name)


def load_train_test_data(file_name='./data/data.csv', split_by='random', ratio=0.25, test_ids=[3], drop_ids=None,
                         is_normalizeX=True, is_normalizeY=False, is_select_features=True, select_features=None, 
                         is_tonumpy=True, is_dropindex=True):
    # load data
    df_features, df_targets = load_data(is_normalizeX=is_normalizeX, is_normalizeY=is_normalizeY, 
                                        is_select_features=is_select_features, select_features=select_features,
                                        file_name=file_name)
    
    # split data
    return split_train_test_data(df_features, df_targets, 
                                 split_by=split_by, ratio=ratio, test_ids=test_ids, drop_ids=drop_ids,
                                 is_tonumpy=is_tonumpy, is_dropindex=is_dropindex)
    
    
def load_train_test_data_outlier(file_name='./data/data_outlier.csv', split_by='random', ratio=0.25, test_ids=[3], drop_ids=None,
                                 is_normalizeX=True, is_normalizeY=False, is_select_features=True, select_features=None, 
                                 is_tonumpy=True, is_dropindex=True):
    # load data
    df_features, df_targets = load_data_outlier(is_normalizeX=is_normalizeX, is_normalizeY=is_normalizeY, 
                                                is_select_features=is_select_features, select_features=select_features,
                                                file_name=file_name)
    
    # split data
    return split_train_test_data(df_features, df_targets, 
                                 split_by=split_by, ratio=ratio, test_ids=test_ids, drop_ids=drop_ids,
                                 is_tonumpy=is_tonumpy, is_dropindex=is_dropindex)
    
    
def split_train_test_data(df_features, df_targets, 
                          split_by='random', ratio=0.25, test_ids=[3], drop_ids=None,
                          is_tonumpy=True, is_dropindex=True):
    # split data
    if split_by == 'random':
        # select design ID
        data_indexes = [ii for ii in xrange(df_features.shape[0])]
        np.random.shuffle(data_indexes)
        data_indexes = data_indexes[0: int(len(data_indexes) * ratio)]
        
        # split dataset
        x_train = df_features[~df_features.index.isin(data_indexes)]
        y_train = df_targets[~df_targets.index.isin(data_indexes)]
        
        x_test = df_features[df_features.index.isin(data_indexes)]
        y_test = df_targets[df_targets.index.isin(data_indexes)]
        
    elif split_by == 'design':
        # select design ID
        design_indexes = df_features['Design_Index'].unique().tolist()
        np.random.shuffle(design_indexes)
        design_indexes = design_indexes[0: int(len(design_indexes) * ratio)]
        
        # split dataset
        x_train = df_features[~df_features['Design_Index'].isin(design_indexes)]
        y_train = df_targets[~df_targets['Design_Index'].isin(design_indexes)]
        
        x_test = df_features[df_features['Design_Index'].isin(design_indexes)]
        y_test = df_targets[df_targets['Design_Index'].isin(design_indexes)]
    
    elif split_by == 'design_give':
        # split dataset
        if drop_ids:
            df_features = df_features[~df_features['Design_Index'].isin(drop_ids)]
            df_targets = df_targets[~df_targets['Design_Index'].isin(drop_ids)]
            
        x_train = df_features[~df_features['Design_Index'].isin(test_ids)]
        y_train = df_targets[~df_targets['Design_Index'].isin(test_ids)]
        
        x_test = df_features[df_features['Design_Index'].isin(test_ids)]
        y_test = df_targets[df_targets['Design_Index'].isin(test_ids)]
    
    elif split_by == 'device_give':
        # split dataset
        if drop_ids:
            df_features = df_features[~df_features['Device_Index'].isin(drop_ids)]
            df_targets = df_targets[~df_targets['Device_Index'].isin(drop_ids)]
            
        x_train = df_features[~df_features['Device_Index'].isin(test_ids)]
        y_train = df_targets[~df_targets['Device_Index'].isin(test_ids)]
        
        x_test = df_features[df_features['Device_Index'].isin(test_ids)]
        y_test = df_targets[df_targets['Device_Index'].isin(test_ids)]
        
    # drop index
    if is_dropindex:
        x_train = x_train.drop(['Design_Index', 'Device_Index'], axis=1)
        y_train = y_train.drop(['Design_Index', 'Device_Index'], axis=1)
        x_test = x_test.drop(['Design_Index', 'Device_Index'], axis=1)
        y_test = y_test.drop(['Design_Index', 'Device_Index'], axis=1)
    
    # to numpy
    if is_tonumpy:
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
    
    # return
    return x_train, y_train, x_test, y_test
    

if __name__ == '__main__':
    df_features, df_targets = load_data()
    # df_features, df_targets = load_data_outlier()
    # x_train, y_train, x_test, y_test = load_train_test_data(split_by='random', ratio=0.25)
    # x_train, y_train, x_test, y_test = load_train_test_data(split_by='design', ratio=0.25)
    # x_train, y_train, x_test, y_test = load_train_test_data(split_by='device', device_ids=3)
    