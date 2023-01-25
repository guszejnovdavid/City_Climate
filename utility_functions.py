#!/usr/bin/env python

### Utility functions for city_climate.ipynb

import pickle
import bz2
import os
import shutil
import numpy as np
from tabulate import tabulate

def pairwise_distance(x,y):
    #Find non-NaN elements and take L2 distance between those, normalized by dimension
    non_nan_ind = np.isfinite(x) & np.isfinite(y)
    N = np.sum(non_nan_ind)
    return np.linalg.norm( x[non_nan_ind]-y[non_nan_ind] )/np.sqrt(N)

def dist_between_keys(key1, key2, df, verbose = False):
    if verbose:
        x = df.loc[key1]
        y = df.loc[key2]
        non_nan_ind = np.isfinite(x) & np.isfinite(y)
        diff = x[non_nan_ind]-y[non_nan_ind]
        ind = np.argsort(-np.abs(diff))
        tab = []
        for i in range(min(15,len(ind))):
            tab.append([np.array(df.columns)[non_nan_ind][ind[i]], diff[ind[i]]])
        print(tabulate(tab,headers =['Climate Parameter','Distance']))
    return pairwise_distance(df.loc[key1], df.loc[key2])
    

def calc_all_pairwise_distances(df):
    array = np.array(df)
    N_item = array.shape[0]
    distances = np.zeros((N_item,N_item)) 
    for i in range(N_item):
        for j in range(i):
            distances[i,j] = pairwise_distance(array[i], array[j]) 
    distances += np.transpose(distances)
    return   distances.astype(np.float32) #use 32 bits to save memory

def label_rel_pops(labels, sort_results=True):
    unique_labels, counts  = np.unique(labels, return_counts =True)
    if sort_results:
        ind = np.argsort(unique_labels)
        unique_labels = unique_labels[ind]; counts = counts[ind]
    return {'Label':unique_labels, 'Count':counts}

def print_label_pop_stat(labels, sort_results=True):
    N_tot = len(labels)
    label_pop_dict = label_rel_pops(labels, sort_results=sort_results)
    for i, label in enumerate(label_pop_dict['Label']):
        print('Label %s:\t %d items,\t %g %% of total'%(label, label_pop_dict['Count'][i], 100*label_pop_dict['Count'][i]/N_tot))    
        
def map_to_koppen_colors(df, default_color=None, verbose=False, skip_rare=True):
    import pandas as pd
    import seaborn as sns
    sns.set()
    #Find the colors of the corresponding clusters on the Köppen climate map
    koppen_pd = pd.read_csv('Koppen_city_climates.csv',encoding = "utf-8" )
    N_cl =  np.max(np.unique(df['label']).astype(int)) + 1
    if default_color is None:
        koppen_color_dict = {cl_id:sns.color_palette(n_colors=N_cl)[int(cl_id)] for cl_id in np.unique(df['label']) }
    else:
        koppen_color_dict = {cl_id:default_color for cl_id in np.unique(df['label']) }
    cl_id_done = []
    for i, (ref_city, color) in enumerate(zip(koppen_pd['Example Key'], koppen_pd['Color'])):
        if (koppen_pd['Rarity'][i]=='No') or (not skip_rare):
            if np.any(df.index == ref_city):
                cl_id = df['label'][np.argmax(df.index == ref_city)]
                if not(cl_id in cl_id_done):
                    if verbose: print(cl_id, koppen_pd['Köppen designation'][i], ref_city, color)
                    koppen_color_dict[cl_id] = color.lower()
                    cl_id_done.append(cl_id)
            else:
                print(ref_city, "missing from dataframe!")
    return koppen_color_dict
    
def pickle_dump(fname, var, use_bz2=False, make_backup=True):
    if make_backup:
        if os.path.exists(fname):
            shutil.copy(fname, fname+'.bak')
    if use_bz2:
        with bz2.BZ2File(fname, 'wb') as pickle_out:
            pickle.dump(var, pickle_out, protocol=2)    
    else:
        with open(fname,'wb') as pickle_out:
            pickle.dump(var, pickle_out)

def pickle_load(fname, use_bz2=False):
    if use_bz2:
        with bz2.BZ2File(fname, 'rb') as pickle_in:
            var = pickle.load(pickle_in, encoding='bytes')
    else:
        with open(fname,'rb') as pickle_in:
            var = pickle.load(pickle_in)
    return var