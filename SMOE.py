#### #### ####
## Module. 
## Implementation Saliency Map Order Equivalence (Mundhenk at al. 2020)
## ## ## ##

import seaborn as sns
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
from scipy.interpolate import CubicSpline
from scipy.signal import argrelextrema, impulse, hamming, exponential, gaussian, boxcar, tukey
from scipy.interpolate import interp1d

### import tensorflow
import tensorflow as tf
tf.random.set_seed(30)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")


def get_feat_maps(trial_signal, CNNnet):
    """
    trial_signal --> tensor, input data
    CNNnet --> trained CNN, model 
    
    RETURN: feature maps for each hidden layer
    """
    
    try:
        layer_list = [layer.output for layer in CNNnet.layers[1:]]
        model_ = tf.keras.models.Model(CNNnet.input, layer_list)
        feat_map= model_.predict(trial_signal)
        return feat_map
    except:
        layer_list = [layer.output for layer in CNNnet.layers[1:]]
        model_ = tf.keras.models.Model(CNNnet.input, layer_list)
        feat_map= model_.predict(trial_signal)
        return feat_map
    
def SMOE(tensor):
    """
    tensor --> hidden layer, activation or feature map. 
    tensor has dimensionality(number samples, lenght map, number features) 
    
    RETURN: SMOE map
    """

    ### set constants
    epsilon = 1e-6
    Nfeats = tensor.shape[2]
    mean_matrix = tensor.mean(axis= 1)
    Z = np.ones((tensor.shape[0], tensor.shape[1]))
    
    ### compute SMOE map
    counter = np.zeros(tensor.shape[1])
    for kk in range(0, Z.shape[0]):
        for jj in range(0 ,Nfeats):
            arglog = mean_matrix[kk, jj]/(tensor[kk, :, jj]+epsilon)
            box =  mean_matrix[kk, jj]*np.log(arglog+epsilon) 
            #print(np.isnan(box).sum())
            counter = counter +box
        
        Z[kk] = counter/Nfeats
        
    return Z


def vanish_feats(X, nfeat= 0):
    
    ### X --> tensor, (samples, lenght-time_domain, time-series-features)
    ### Set one time-series feature to zero-value
    
    Y = np.zeros(X.shape)
    Y[:, :, nfeat] = X[:, :, nfeat]
    return Y

def simply_apodization(X, droprate= 10e-2):
    w_apo = tukey(X.shape[0], alpha= droprate)
    return w_apo*X

def upsampling_SMOE(smoe, final_lenght= 1024):
    
    """
    #######################################################
    ### Upsampling on SMOE maps
    ### smoe --> saliency map, matrix, as coopmuted by SMOE function
    ### final_lenght --> desidred lenght of SMOE map 
    ##########################################################
    """
    
    ### set knots
    hknot = int(final_lenght/smoe.shape[1])
    mknot = smoe.shape[1]+2
    knots = np.linspace(0, final_lenght, mknot)[1:mknot-1] 
    
    ## convolutinal window 
    tau = int(final_lenght/smoe.shape[1])
    tau += tau%2
    W = hamming(tau)/hamming(tau).sum()
   
    ### ### ###
    ## upsamling via zero-spline interplation, then use convolution.
    Z = np.empty((smoe.shape[0], final_lenght)) #(numero mappe, lunghezza effettiva della mappa)
    T = np.arange(0, final_lenght+tau-1)
    for kk in range(0, smoe.shape[0]):
        piecewise = interp1d(x= knots, 
                             y= smoe[kk], 
                             kind= 'zero', 
                             bounds_error= False, 
                             fill_value= 'extrapolate')
        Z[kk] = np.convolve(piecewise(T), W, mode= 'valid')
    return Z

def gamma_SMOE(X):
    
    #### ####
    ## X --> SMOE map 
    ## RETURN : apply monotone function to single SMOE map (gaussian cdf)
   
    m0 = X.mean()
    s0 = X.std()
    gamma = norm.cdf(X, loc= m0, scale= s0)
    return gamma

def Gamma_SMOE(X):
    
    """
    #################################################
    ### Combined SMOE map (on all hidden layers)
    ### X --> SMOE ap (as matrix, number_layersXlenght_smoe_map)
    ### 
    
    """
    
    W = np.ones(X.shape[0])
    W = W/W.sum()
    #W = W[0:X.shape[0]]
    
    ###
    Y = X.copy()
    for kk in range(0, Y.shape[0]):
        Y[kk] = gamma_SMOE(X[kk])*W[kk]
    
    return  Y.sum(axis= 0).ravel()
    
###########
### ### ####
def plot_SMOEs(conv_feat_maps, strides_first_layer= 1):
    
    ####
    ### plot all smoes
    ## conv_feat_maps --> lista con le feat_map convoluzionali (after activation)
    ## strides 1st layer --> strides rispetto al primo layer per ricostruire l'asse temporale.
    ####
    
    ### ### ###
    kolors = sns.color_palette()
    lenght_smoe = conv_feat_maps[0].shape[1]*strides_first_layer    
    
    ### ### ###
    ###plot ###
    nmax_layers = len(conv_feat_maps)
    plt.figure(1, figsize=(3*nmax_layers, 4*nmax_layers))
    for kk in range(0, nmax_layers):
        plt.subplot(nmax_layers, 1, kk+1)
        single_smoe = SMOE(conv_feat_maps[kk])
        plt.plot(np.linspace(0, lenght_smoe, single_smoe.shape[1]), single_smoe.ravel(),
                 c= kolors[kk], label= 'SMOE_layer_'+str(kk+1))
        plt.grid(True)
        plt.ylabel('SMOE')
        plt.xlabel('Time-Series Localization [pixels]')
        plt.legend()

    plt.show()
    
    
def SMOEcombined(conv_feat_maps, lenght_smoe = 64, weights= 'standard'):
    
    """
    ### Compute combined SMOE map
    ### ###
    conv_feat_map --> LIST with activated hidden maps (this list represents one single sample)
    lenght_smoe  --> lenght of the desired combined SMOE map.
    weights --> how to weight each SMOE map. _standard_ equally likely hidden layers; _exp_ the more the deepness, the more the relevance is
    """
    
    ## initialization
    N0 = conv_feat_maps[0].shape[0]
    smoes_list = np.zeros((N0, len(conv_feat_maps), lenght_smoe))
    for kk in range(0, len(conv_feat_maps)):
        smoe_0 = SMOE(conv_feat_maps[kk])
        upsmoe_ = upsampling_SMOE(smoe= smoe_0, final_lenght= lenght_smoe)
        smoe_ = gamma_SMOE(upsmoe_)
        smoes_list[:, kk, :] = smoe_
        
        
    ################################
    if weights == 'standard':
        W = np.arange(0, len(conv_feat_maps), 1)
        W = W/W.sum()
    elif weights == 'exp':
        W = exponential(len(conv_feat_maps), center= len(conv_feat_maps)-1, tau= len(conv_feat_maps)/np.log(2), sym= False)
        #W = np.ones(len(smoes_list))
        W = W/W.sum()
    
    ### dimension manipulation 
    Z = np.zeros(smoes_list.shape)
    for kk in range(0, smoes_list.shape[1]):
        Z[:, kk, :] = smoes_list[:, kk, :]*W[kk]
    
    
    return np.clip(np.sum(Z, axis= 1), a_min= 0, a_max= 1)
