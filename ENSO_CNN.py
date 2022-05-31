####
# ENSO_CNN. 
###
import pandas as pd
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

###
from scipy.signal import argrelextrema, impulse, hamming
from scipy.special import jv, j0, j1, gamma, eval_hermite

##
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import KernelDensity

## TENSOR FLOW
import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.callbacks import Callback


    
def scaler(data, mm = 0, MM = 0):
    Z = data.copy()
    return (Z-mm)/(MM-mm)

    
def max_deepness(wamplitude, ks):
    """
    compute the max deepnes of CNN (with poolsize 2),  given the amplitude and the kernel_size 
    """
    return int(np.log2((wamplitude-ks+1)/ks))
    
#### CNN model
def CNN_(Xtrain, Ytrain,
         Xtest, Ytest, 
        nfilt = 10, 
        ks = 64, 
        pooling= 2, 
        strides= 1, 
        DropOut = .5,
        deepness = 1,
        pred_stddev = 0,  
        actv1 = 'relu',
        padding = 'same',
        lr= 1e-3, 
        Verbose= 0, 
        Epochs= 100, 
        batch_size = 100,
        patience = 20, 
        winit = 2e-1, 
        platt = True, 
        bias = False, 
        display_summary= False):
                          
    
    """
    Convolutinal Neural Network 1D. 
    Define model and fit it.
    
    Xtrain, Ytrain, Xtest, Ytest --> Input data
    nfilt -> number of filters (per convolutional layer)
    ks -> kernel_size   
    pooling -> pooling size in MaxPooling Layer
    strides -> strides side, default 1; no stride is applied 
    DropOut -> Dropout rate
    deepness -> Number of hidden layers (i.e Conv1D+Activation+Maxpooling+Dropout)
    pred_stddev -> standard deviation of additive zero-mean white noise, applied on the prediction layer (Dense). Defalut 0; no noise is applied.  
    actv1 -> activation function, Use RELU
    padding -> padding on convolutional layers, Use valid
    lr -> learing rate (during learning phase)
    Verbose -> Print trianing progress during the learning phase, can be either 0 or 1 (or 2). 0 means no print.
    Epochs  -> Maximum number of Epochs 
    batch_size -> batch size at each epoch
    patience -> tolerance value after which stopping the learning phase (used under Early Stopping Criterion)  
    winit -> Kernel values of initialization 
    platt -> whether apply Platt scaling (Default False) 
    bias -> Fit bias terms (Default False) 
    display_summary -> Displayies CNN-1D architecture (Default False)
    """
    
    ## initialize weights
    Winit_ = tf.keras.initializers.TruncatedNormal(mean=0, stddev=winit, seed=30) 
    
    ## inizitialize bias terms
    Binit_ = 'zeros'
    
    ### Define Input
    Input_Shape = (Xtrain.shape[1], Xtrain.shape[2])
    Inputs = tf.keras.Input(shape=Input_Shape)
    
    ## 1st Block (Conv1d+activation+pooling+dropout)
    X = tf.keras.layers.Conv1D(filters= nfilt,
                          kernel_size= ks,
                          padding= padding,
                          activation= 'linear',
                          strides= strides, 
                          kernel_initializer= Winit_,
                          use_bias = bias,
                          bias_initializer=Binit_)(Inputs)
    X = tf.keras.layers.Activation(actv1)(X)
    X = tf.keras.layers.MaxPooling1D(pooling, padding= 'same')(X)
    X = tf.keras.layers.Dropout(rate= DropOut)(X)
    
    ## all further blocks
    for KK in range(0, deepness-1):
        
        X = tf.keras.layers.Conv1D(filters= nfilt,
                          kernel_size= ks,
                          padding= padding,
                          activation= 'linear',
                          kernel_initializer= Winit_,
                          bias_initializer=Binit_,
                          use_bias = bias,
                          strides= strides)(X)
        X = tf.keras.layers.Activation(actv1)(X)
        X = tf.keras.layers.MaxPooling1D(pooling, padding= 'same')(X)
        X = tf.keras.layers.Dropout(rate= DropOut)(X)
    
        
    ### final encoding -- flattened array
    X = tf.keras.layers.Flatten()(X)
    
    ## USE Platt scaling, then make final prediction (Dense Layer + sigmoidal activation)
    if platt:
        X = tf.keras.layers.Dense(units= nfilt, activation= 'linear', use_bias= bias)(X)
        X = tf.keras.layers.Dense(units= 1, activation= 'sigmoid', 
                                  use_bias= bias, 
                                  kernel_initializer= Winit_, 
                                  bias_initializer= 'zeros')(X)
        final_pred = tf.keras.layers.GaussianDropout(rate= pred_stddev)(X)
    
    ## DON'T USE Platt scaling, then make final prediction (Dense Layer + sigmoidal activation)
    else:
        X = tf.keras.layers.Dense(units= 1, activation= 'sigmoid', 
                                           use_bias = bias, 
                                           kernel_initializer= Winit_, 
                                  bias_initializer= 'zeros')(X)
        final_pred = tf.keras.layers.GaussianDropout(rate= pred_stddev)(X)
        

    ### Define as Keras Model
    CNN_ = tf.keras.models.Model(inputs= Inputs, outputs= final_pred)
    
    ###Display Architecture of the Model
    if display_summary == True:
        print(CNN_.summary())
     
    ###Optimizers
    adam = tf.keras.optimizers.Adam(lr= lr)
    sgd = tf.keras.optimizers.SGD(learning_rate= lr)
    
    ##Loss Function (Binary Cross-Entropy)
    bce = tf.keras.losses.BinaryCrossentropy()
    
    ### Define EarlyStopping Criterion as Callback
    early__ = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience)
    
    ### Compile the model
    CNN_.compile(optimizer= adam, loss = bce)
    
    #### #### #### ####
    ### FIT THE MODEL, run .fit 
    history = CNN_.fit(Xtrain, 
                       Ytrain, 
                       validation_data = (Xtest, Ytest),
                       verbose= Verbose, 
                       epochs= Epochs, 
                       batch_size= batch_size, 
                       callbacks = [early__])
    
    return CNN_, history
    
#####visualization results
def ccd_plot(Ypred, Ytrue, nbins= 21):
    
    """
    CCD_plot -> Plot the Class Conditional Distributions
    Ypred -> Either Probabilistic score or the evaluation of the decisional function (values must lie between 0 and 1)
    YTrue -> True Labels
    nbins -> Number of bins in the final histograms
    """
    
    
    ###Evaluate ROC-AUC (ROC Area Under the Curve)
    auc = roc_auc_score(Ytrue, Ypred)
    print('*****')
    print('AUC')
    print(auc.round(2))

    ### stratify the score value per class
    YT = Ypred[Ytrue == 1].astype(float)
    YF = Ypred[Ytrue == 0].astype(float)

    #### #### ####
    print('*****')
    print('*****')
    #### ##### ####
    
    ### POSITIVE INSTANCES -- HISTOGRAM
    plt.figure(1, figsize= (12, 12))
        #plt.title('Class Conditional Distirbution--'+name_cov)
    plt.subplot(2, 2, 1)
    bbins = bins= np.linspace(0, 1, nbins, endpoint= True)
    plt.hist(YT, weights= np.ones_like(YT)/np.shape(YT)[0], 
             color= 'tab:blue', bins= bbins, 
             label= 'Positive Density')
    #hist_pos = np.histogram(YT, weights= np.ones_like(YT)/np.shape(YT)[0], bins= bbins, density= False)
    hist_pos = np.histogram(YT, bins= bbins, density= True)
    plt.xlabel('NN score')
    plt.ylabel('Density Estimation')
    plt.grid(True)
    plt.legend()
    
    #### NEGATIVE INSTANCES - HISTOGRAM
    plt.subplot(2, 2, 2)
    plt.hist(YF, weights= np.ones_like(YF)/np.shape(YF)[0], 
             color= 'tab:red', bins= bbins, 
             label= 'Negative Density')
    #hist_neg = np.histogram(YF, weights= np.ones_like(YF)/np.shape(YF)[0], bins= bbins)
    hist_neg = np.histogram(YF, bins= bbins, density= True)
    plt.xlabel('NN score')
    plt.ylabel('Density Estimation')
    plt.grid(True)
    plt.legend()
        
    print('*** *** ***')
    print('@@@ @@@ @@@')
    print('*** *** ***')
   
    ### ROC-AUC, estimate ROC, then estimate AUC
    fpr, tpr, eps = roc_curve(Ytrue, Ypred)
    plt.subplot(2, 2, 3)
    plt.plot(fpr, tpr, c= 'tab:green', label= 'ROC: Area('+str(np.round(auc, 2))+')')
    plt.plot(np.arange(0, 1.1, .1), np.arange(0, 1.1, .1), '--', c= 'black')
    plt.xlabel('False Positive Ratio')
    plt.ylabel('True Positive Ratio')
    plt.legend()
    plt.grid(True)
    
    #### KERNEL DENSITY ESTIMATION for both the classes
    plt.subplot(2, 2, 4)
    scores_ = bbins
    bw = np.diff(scores_).mean()
    ### Positive Instances
    kde = KernelDensity(kernel='tophat', bandwidth= bw).fit(YT.reshape(-1, 1))
    density_ = np.exp(kde.score_samples(scores_.reshape(-1, 1)))
    etamax, etamin = hist_pos[0].max(), hist_pos[0].min()
    a = (etamax-etamin)/(density_.max()-density_.min())
    b = (density_.max()*etamin-density_.min()*etamax)
    density_ = density_*a+b
    plt.plot(scores_, density_, c= 'tab:blue', ls= '--', label= 'KDE_positive')
    #### Negative Instances
    kde = KernelDensity(kernel='tophat', bandwidth= bw).fit(YF.reshape(-1, 1))
    scores_ = bbins
    density_ = np.exp(kde.score_samples(scores_.reshape(-1, 1)))
    etamax, etamin = hist_neg[0].max(), hist_neg[0].min()
    a = (etamax-etamin)/(density_.max()-density_.min())
    b = (density_.max()*etamin-density_.min()*etamax)
    density_ = density_*a+b
    plt.plot(scores_, density_, c= 'tab:red', ls= '--', label= 'KDE_negative')
    plt.xlabel('NN score')
    plt.ylabel('Log Density Estimation')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)    
    
    ### PLOT, make the final plot
    plt.show()
    return 
