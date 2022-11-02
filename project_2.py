#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 17:06:01 2022

@author: eddyvonmatt
"""

###-------------------------------------------------------------------------------------------------------------------
#         PREDICTING THE REFRACTIVE INDEX OF GLASS BASED ON ITS CHEMICAL COMPONENTS
###-------------------------------------------------------------------------------------------------------------------

import csv
#from matplotlib.pyplot import (figure, subplot, plot, hist, show, boxplot, matshow, yticks, xticks, ylabel, ylim, xlabel, legend, title)
import numpy as np
# from scipy import stats
import pandas as pd
# import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import model_selection
import torch
import project_2_lib

###-------------------------------------------------------------------------------------------------------------------
#          PART A - Regularized Linear Regression / CV - find optimal lambda
###-------------------------------------------------------------------------------------------------------------------

###-------------------------------------------------------------------------------------------------------------------
#          import data
###-------------------------------------------------------------------------------------------------------------------
attributeNames = [ "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
file = open('glass.csv')
csvreader = csv.reader(file, delimiter=',')
data_list = []
for row in csvreader:
    data_list.append(row)
file.close()
numpy_array_string = np.array([np.array(x) for x in data_list])
X = numpy_array_string.astype(float)
y = X[:, 1]
X = X[:, 2:-1]
N,M = X.shape

###-------------------------------------------------------------------------------------------------------------------
#          normalize data and offset 
###-------------------------------------------------------------------------------------------------------------------
# maybe try that later as in the provided code
# mu_train = np.mean(X, 0)
# sigma_train = np.std(X, 0)

X = X - np.ones((N, 1))*X.mean(0)
X = X*(1/np.std(X,0))

X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

###-------------------------------------------------------------------------------------------------------------------
#          cross validation parameters
###-------------------------------------------------------------------------------------------------------------------
K = 10
CV = model_selection.KFold(K, shuffle=True)

###-------------------------------------------------------------------------------------------------------------------
#          range of lambda models
###-------------------------------------------------------------------------------------------------------------------
lambdas = np.power(10.,range(-5,9))

Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

for lambda_value in lambdas:
    k=0
    for train_index, test_index in CV.split(X,y):
        # extract training and test set for current CV fold
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        
        # Estimate weights for the optimal value of lambda, on entire training set
        lambdaI = lambda_value * np.eye(M)
        lambdaI[0,0] = 0 # Do no regularize the bias term
        
        w_rlr = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        
        # Compute mean squared error with regularization with lambda
        Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr).sum(axis=0)/y_train.shape[0]
        Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr).sum(axis=0)/y_test.shape[0]
        
        # Estimate weights for unregularized linear regression, on entire training set
        w_noreg = np.linalg.solve(XtX,Xty).squeeze()
        
        # Compute mean squared error without regularization
        Error_train[k] = np.square(y_train-X_train @ w_noreg).sum(axis=0)/y_train.shape[0]
        Error_test[k] = np.square(y_test-X_test @ w_noreg).sum(axis=0)/y_test.shape[0]
        k += 1
    print(sum(Error_train_rlr)/len(Error_train_rlr))
    print(sum(Error_test_rlr)/len(Error_test_rlr))
    print(sum(Error_train)/len(Error_train))
    print(sum(Error_test)/len(Error_test))
    print('Weights in last fold:')
    for m in range(M):
        print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m],10)))


###-------------------------------------------------------------------------------------------------------------------
#          PART B - ANN Regression vs. Regularized Regression vs. Baseline - Two Level CV
###-------------------------------------------------------------------------------------------------------------------


###-------------------------------------------------------------------------------------------------------------------
#          load data
###-------------------------------------------------------------------------------------------------------------------
attributeNames = [ "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
file = open('glass.csv')
csvreader = csv.reader(file, delimiter=',')
data_list = []
for row in csvreader:
    data_list.append(row)
file.close()
numpy_array_string = np.array([np.array(x) for x in data_list])
X = numpy_array_string.astype(float)
y = X[:, [1]]
X = X[:, 2:-1]
N,M = X.shape

###-------------------------------------------------------------------------------------------------------------------
#          normalize data - QUESTION - why necessary? - will it affect my regression? - if yes apply it later
###-------------------------------------------------------------------------------------------------------------------
# X = stats.zscore(X) 

X = X - np.ones((N, 1))*X.mean(0)
X = X*(1/np.std(X,0))

# Add offset attribute - only for RLR
X_rlr = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames_rlr = [u'Offset']+attributeNames # not really used
M_rlr = M+1

###-------------------------------------------------------------------------------------------------------------------
#          parameters for neural network classifier - QUESTION - what exactly is n_replicates for?
###-------------------------------------------------------------------------------------------------------------------
n_hidden_units = range(1, 15)
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 10000

###-------------------------------------------------------------------------------------------------------------------
#          parameters for regularized linear regression
###-------------------------------------------------------------------------------------------------------------------
lambda_values = np.power(10.,range(-5,9))

###-------------------------------------------------------------------------------------------------------------------
#          inner and outer K-fold crossvalidation
###----------------------------------------------------------------------------------------------------------------
K = 2
CV = model_selection.KFold(K, shuffle=True)
K_inner = 2
CV_inner = model_selection.KFold(K_inner, shuffle=True)

###-------------------------------------------------------------------------------------------------------------------
#          setup figure for display of learning curves and error rates in fold
###-------------------------------------------------------------------------------------------------------------------
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
# colors for learning curve plots
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']


###-------------------------------------------------------------------------------------------------------------------
#         CV outer loop
###-------------------------------------------------------------------------------------------------------------------

errors_nn = [] # make a list for storing generalizaition error in each loop
optimal_n_hidden_units = [] # make a list for storing optimal layer number in each loop

errors_rlr = [] # make a list for storing generalizaition error in each loop
optimal_lambdas = [] # make a list for storing optimal lambda value in each loop

errors_no_f_lg = [] # make a list for storing generalizaition error in each loop

for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
    print('\nCrossvalidation fold outer: {0}/{1}'.format(k+1,K))    
    
    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.Tensor(X[train_index,:])
    y_train = torch.Tensor(y[train_index])
    X_test = torch.Tensor(X[test_index,:])
    y_test = torch.Tensor(y[test_index])
    
    X_train_rlr = X_rlr[train_index]
    y_train_rlr = y[train_index].flatten().T
    X_test_rlr = X_rlr[test_index]
    y_test_rlr = y[test_index].flatten().T
    
    
    # neural network - inner errors for each hidden layer in the current fold
    inner_errors_nn = np.array([])
    
    # regularized_linear_regression - inner errors for each lambda in the current fold
    inner_errors_rlr = np.array([])
    
    ###-------------------------------------------------------------------------------------------------------------------
    #         CV inner loop
    ###-------------------------------------------------------------------------------------------
    
    for (k_inner, (train_index_inner, test_index_inner)) in enumerate(CV_inner.split(X_train,y_train)): 
        print('\nCrossvalidation fold inner: {0}/{1}'.format(k_inner+1,K))    

        # Extract training and test set for current inner CV fold, convert to tensors
        X_train_inner = torch.Tensor(X[train_index_inner,:])
        y_train_inner = torch.Tensor(y[train_index_inner])
        X_test_inner = torch.Tensor(X[test_index_inner,:])
        y_test_inner = torch.Tensor(y[test_index_inner])
        
        X_train_inner_rlr = X_rlr[train_index]
        y_train_inner_rlr = y[train_index].flatten().T
        X_test_inner_rlr = X_rlr[test_index]
        y_test_inner_rlr = y[test_index].flatten().T
        
        ###-------------------------------------------------------------------------------------------------------------------
        #         neural network with regression - inner fold - maybe function
        ###-------------------------------------------------------------------------------------------------------------------
        inner_errors_n_hidden_unit = []
        for n_hidden_unit in n_hidden_units:
            # Define the model
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, n_hidden_unit), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(n_hidden_unit, 1), # n_hidden_units to 1 output neuron
                                # no final transfer function, i.e. "linear output"
                                )
            loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
            
            # Train the net on training data
            net_inner, final_loss_inner, learning_curve_inner = project_2_lib.train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train_inner,
                                                               y=y_train_inner,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)
        
            # Determine estimated class labels for test set
            y_test_est_inner = net_inner(X_test_inner)
            
            # Determine errors and errors
            se_inner = (y_test_est_inner.float()-y_test_inner.float())**2 # squared error
            mse_inner = (sum(se_inner).type(torch.float)/len(y_test_inner)).data.numpy()[0] #mean
            
            # store error rate for current CV fold for all the numbers of hidden units
            inner_errors_n_hidden_unit.append(mse_inner)
        if len(inner_errors_nn) == 0:
            inner_errors_nn = inner_errors_n_hidden_unit
        else:
            inner_errors_nn = np.vstack([inner_errors_nn, inner_errors_n_hidden_unit])   
           
        ###-------------------------------------------------------------------------------------------------------------------
        #        regularized_linear_regression - inner fold - maybe function
        ###-------------------------------------------------------------------------------------------------------------------
        inner_errors_lambda = []
        for lambda_value in lambda_values:
            Xty = X_train_inner_rlr.T @ y_train_inner_rlr
            XtX = X_train_inner_rlr.T @ X_train_inner_rlr
            
            # Estimate weights for the optimal value of lambda, on entire training set
            lambdaI = lambda_value * np.eye(M_rlr)
            lambdaI[0,0] = 0 # Do no regularize the bias term
            
            w_inner_rlr = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
            
            # Compute mean squared error with regularization with lambda
            Error_test_rlr = np.square(y_test_inner_rlr-X_test_inner_rlr @ w_inner_rlr).sum(axis=0)/y_test_inner_rlr.shape[0]
            inner_errors_lambda.append(Error_test_rlr)
        if len(inner_errors_rlr) == 0:
            inner_errors_rlr = inner_errors_lambda
        else:
            inner_errors_rlr = np.vstack([inner_errors_rlr, inner_errors_lambda])   
    
    print('\nCrossvalidation fold outer: {0}/{1}'.format(k+1,K))    

    ###-------------------------------------------------------------------------------------------------------------------
    #         neural network with regression - outer fold - error computation
    ###-------------------------------------------------------------------------------------------------------------------    
    average_error_per_hu = np.sum(inner_errors_nn, axis=0)/K_inner
    optimal_n_hidden_unit_average_error = np.min(average_error_per_hu)
    optimal_n_hidden_unit = n_hidden_units[np.argmin(average_error_per_hu)]
        
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, optimal_n_hidden_unit), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(optimal_n_hidden_unit, 1), # n_hidden_units to 1 output neuron
                        # no final transfer function, i.e. "linear output"
                        )
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
    
    # Train the net on training data
    net, final_loss, learning_curve = project_2_lib.train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
        
    # Determine estimated class labels for test set
    y_test_est = net(X_test)
    
    # Determine errors and errors
    se = (y_test_est.float()-y_test.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
    
    # return
    errors_nn.append(mse) # store error rate for current CV fold 
    optimal_n_hidden_units.append(optimal_n_hidden_unit)
    
    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')
    #summaries_axes[0].set_ylim([0, 0.6])
    
    ###-------------------------------------------------------------------------------------------------------------------
    #         regularized linear regression - outer fold - error computation
    ###-------------------------------------------------------------------------------------------------------------------
    
    average_error_per_lambda = np.sum(inner_errors_rlr, axis=0)/K_inner
    optimal_lambda_average_error = np.min(average_error_per_lambda)
    optimal_lambda_value = lambda_values[np.argmin(average_error_per_lambda)]
    optimal_lambdas.append(optimal_lambda_value)
    
    Xty = X_train_rlr.T @ y_train_rlr
    XtX = X_train_rlr.T @ X_train_rlr
    
    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = optimal_lambda_value * np.eye(M_rlr)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    
    w_rlr = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    
    # Compute mean squared error with regularization with lambda
    Error_test_rlr = np.square(y_test_rlr-X_test_rlr @ w_rlr).sum(axis=0)/y_test_rlr.shape[0]
    errors_rlr.append(Error_test_rlr)
    
    ###-------------------------------------------------------------------------------------------------------------------
    #         baseline - outer fold - error computation
    ###-------------------------------------------------------------------------------------------------------------------
    
    # Compute mean squared error - baseline
    Error_test_baseline = np.square(y_test_rlr-np.mean(y_train_rlr)).sum(axis=0)/y_test_rlr.shape[0]
    errors_no_f_lg.append(Error_test_baseline)
        


###-------------------------------------------------------------------------------------------------------------------
#         statistical analysis TBC
###-------------------------------------------------------------------------------------------------------------------
 
d = {'outer fold' : range(1,len(optimal_n_hidden_units)+1), 'h_i': optimal_n_hidden_units, 'e_i1': errors_nn, 'lambda_i': optimal_lambdas, 'e_i2': errors_rlr, 'e_i3': errors_no_f_lg}
performance_table = pd.DataFrame(d)

# ANN vs. linear regression; ANN vs. baseline; linear regression vs. baseline - maybe abs?
r_ann_lrl = [y - x for y, x in zip(errors_nn, errors_rlr)]
r_ann_baseline = [y - x for y, x in zip(errors_nn, errors_no_f_lg)]
r_lrl_baseline = [y - x for y, x in zip(errors_rlr, errors_no_f_lg)]

# Initialize parameters and run test appropriate for setup II
alpha = 0.05
rho = 1/K


###-------------------------------------------------------------------------------------------------------------------
#         neural network with regression - not relevant diagrams - use later
###-------------------------------------------------------------------------------------------------------------------
 
# Display the MSE across folds
summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors_nn)), color=color_list)
summaries_axes[1].set_xlabel('Fold')
summaries_axes[1].set_xticks(np.arange(1, K+1))
summaries_axes[1].set_ylabel('MSE')
summaries_axes[1].set_title('Test mean-squared-error')
    
print('Diagram of best neural net in last fold:')
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(net[i]) for i in [1,2]]
project_2_lib.draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

#Print the average classification error rate
print('\nEstimated generalization error of neural network, RMSE: {0}'.format(round(np.sqrt(np.mean(errors_nn)), 4)))

# When dealing with regression outputs, a simple way of looking at the quality
# of predictions visually is by plotting the estimated value as a function of 
# the true/known value - these values should all be along a straight line "y=x", 
# and if the points are above the line, the model overestimates, whereas if the
# points are below the y=x line, then the model underestimates the value
plt.figure(figsize=(10,10))
y_est = y_test_est.data.numpy(); y_true = y_test.data.numpy()
axis_range = [np.min([y_est, y_true])-1,np.max([y_est, y_true])+1]
plt.plot(axis_range,axis_range,'k--')
plt.plot(y_true, y_est,'ob',alpha=.25)
plt.legend(['Perfect estimation','Model estimations'])
plt.title('Refractive Index: estimated versus true value (for last CV-fold)')
plt.ylim([1.50, 1.54]); plt.xlim(axis_range)
plt.xlabel('True value')
plt.ylabel('Estimated value')
plt.grid()
plt.show()