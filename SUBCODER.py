import pandas as pd
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.lines as mlines
import sys
from datetime import datetime as dt
from scipy.stats import skew, kurtosis, sem, mode, iqr, pearsonr
from scipy.special import erf
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.signal import hamming, boxcar, butter, filtfilt, welch, tukey, detrend, gaussian, butter
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA

##
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, silhouette_score, roc_curve, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.cluster import SpectralClustering, KMeans

### TENSOR FLOW
#import tensorflow as tf
#tf.random.set_seed(30)
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import ENSO_ccn as nn
#import SMOE as smoe
import raw_paths
import ENSOutils as utils

#####
# SUBBANDER


#### #### #### #### ####
def subcoding_iterative_filtering(X, standardize= True, kind= 'dyadic', nbands= 8, btype='lowpass'):

    """ 
    X--> Time-Series to split into sub-bands
    standardize --> whether to standardize the signal
    btype --> whether "lowpass" or 'bandpass'
    kind --> dyadic or equal split
    nbands --> number of desidered subbands
    
    RETURN : subcode on the frequency domain
    """    
    
    if kind == 'dyadic':
        epsilon0 = .5/X.shape[0]
        nbands= int(-np.log2(epsilon0))
        norm_fs = 1/np.power(2, np.arange(0, nbands, 1))
        norm_fs[0] = 1-epsilon0
        norm_fs[-1] = epsilon0
    elif kind== 'equal':
        epsilon0 = .5/X.shape[0]
        norm_fs = np.linspace(1-epsilon0, epsilon0, nbands)
    
    
    if standardize:
        Xcopy = X.copy()
        Xcopy = Xcopy-Xcopy.mean() 
    else:
        Xcopy = X.copy()
    
    Npad= int(X.size)
    SS = np.pad(Xcopy, pad_width=(Npad, Npad), mode='edge')
    Zfiltered = np.zeros((SS.shape[0], nbands))
    for kk in range(0, nbands):
        campina_filter = butter(N= 3, Wn= norm_fs[kk], btype= btype, analog= False)
        campina_signal = filtfilt(b= campina_filter[0], a= campina_filter[1], x= SS)
        Zfiltered[:, kk-1] = campina_signal
    
    return Zfiltered[Npad:2*Npad]


def subcode_number_of_bands(lenght):
    """
    Returns number of bands given the lenght of the Time-Series
    """
    epsilon0 = .5/lenght
    nbands= int(-np.log2(epsilon0))
    return nbands

def spectral_band_per_layer(lenght):

    """
    Returns spetracl bands given the lenght of the Time-Series (dyadic sub-coding)
    """
    epsilon0 = .5/lenght
    nbands= int(-np.log2(epsilon0))
    norm_fs = 1/np.power(2, np.arange(0, nbands, 1))
    norm_fs[0] = 1-epsilon0
    norm_fs[-1] = epsilon0
    
    print('**** **** ****')
    print('SPECTRAL BAND PER LAYER [PERIODS]')
    for kk in range(1, nbands):
        f1, f2 = 2/norm_fs[kk-1], 2/norm_fs[kk]
        print('LAYER_'+str(kk)+':', f1.round(2), f2.round(2))
    print('**** **** ****')
    return


def inst_subbands_iterfilt(X, standardize= True, kind= 'dyadic', nbands= 8, btype= 'highpass'):
    
   """
    X --> instance (samples, time-domain, ts_features)
    ... same args like "subcoding_iterative_filtering"...
    
   RETURN: split each ts_features into subbands, tensor with dimensions (samples, time-domain, ts_features, number sub-band) 
   """
    
    if kind == 'dyadic':
        nbands = subbander_number_of_bands(X.shape[1])
    
    Npats, time, nfeats = X.shape
    Zimfs = np.zeros((Npats, time, nfeats, nbands))
    for npats in range(0, Npats):
        for jj in range(0, nfeats):
            Zimfs[npats, :, jj, :] = subcoding_iterative_filtering(X[npats, :, jj], standardize, kind, nbands, btype) 
        
    return Zimfs
