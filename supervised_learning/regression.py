'''
REGRESSION : Predicts a value based on labeled data. For example, 
                we can predict house prices depending on some feautures
'''


###############    SOME SKLEARN SUPERVISED LEARNING AND REGRESSION MODELS    ####################


# 1. LINEAR REGRESSION MODELS
# Model the relationship between input features and a
# continuous target using linear equations.

# LinearRegression
# Module: sklearn.linear_model
# Fits a linear equation by minimizing the sum of squared
# residuals (Ordinary Least Squares).

# Ridge
# Module: sklearn.linear_model
# Linear regression with L2 regularization to reduce
# overfitting and handle multicollinearity.

# Lasso
# Module: sklearn.linear_model
# Linear regression with L1 regularization. Can shrink
# some coefficients to zero, performing feature selection.

# ElasticNet
# Module: sklearn.linear_model
# Combines L1 and L2 regularization to balance feature
# selection and coefficient shrinkage.

# SGDRegressor
# Module: sklearn.linear_model
# Trains linear regression models using Stochastic
# Gradient Descent. Suitable for large datasets.

# BayesianRidge
# Module: sklearn.linear_model
# Bayesian linear regression with L2-like regularization
# and probabilistic estimation of model parameters.

# HuberRegressor
# Module: sklearn.linear_model
# Robust linear regression that reduces the influence
# of outliers using the Huber loss function.

# RANSACRegressor
# Module: sklearn.linear_model
# Robust regression that identifies inliers and fits
# a model while ignoring potential outliers.

# TheilSenRegressor
# Module: sklearn.linear_model
# Robust linear regression based on combinations of
# subsets of observations.


# 2. POLYNOMIAL REGRESSION
# Models nonlinear relationships by adding polynomial
# combinations of the original features.

# PolynomialFeatures + LinearRegression
# Modules: sklearn.preprocessing + sklearn.linear_model
# Generates polynomial features and fits a linear model
# to capture nonlinear relationships.

# SplineTransformer + LinearRegression
# Modules: sklearn.preprocessing + sklearn.linear_model
# Generates spline basis features to model smooth,
# flexible nonlinear relationships.


# 3. TREE-BASED REGRESSION MODELS
# Split the dataset into regions using decision rules
# to predict continuous numerical values.

# DecisionTreeRegressor
# Module: sklearn.tree
# Predicts numerical values using a tree of decision
# rules based on feature values.

# ExtraTreeRegressor
# Module: sklearn.tree
# Uses randomized feature splits to construct a
# decision tree for regression.


# 4. ENSEMBLE REGRESSION MODELS
# Combine multiple estimators to improve prediction
# accuracy, stability and generalization.

# RandomForestRegressor
# Module: sklearn.ensemble
# Combines multiple decision trees trained on different
# bootstrap samples and averages their predictions.

# ExtraTreesRegressor
# Module: sklearn.ensemble
# Combines highly randomized decision trees to reduce
# variance and improve generalization.

# BaggingRegressor
# Module: sklearn.ensemble
# Trains multiple regressors on different bootstrap
# samples and averages their predictions.

# AdaBoostRegressor
# Module: sklearn.ensemble
# Sequentially trains weak regressors, giving more
# importance to observations with larger prediction errors.

# GradientBoostingRegressor
# Module: sklearn.ensemble
# Builds sequential regression models that correct
# the residual errors of previous models.

# HistGradientBoostingRegressor
# Module: sklearn.ensemble
# Optimized gradient boosting using histogram-based
# learning for efficient training on large datasets.

# VotingRegressor
# Module: sklearn.ensemble
# Combines predictions from multiple regression models
# by averaging their outputs.

# StackingRegressor
# Module: sklearn.ensemble
# Combines several regressors using a final estimator
# that learns how to integrate their predictions.


# 5. SUPPORT VECTOR REGRESSION (SVR)
# Predict continuous values by fitting a function within
# a defined error margin.

# SVR
# Module: sklearn.svm
# Support Vector Regression. Supports linear and nonlinear
# kernels such as RBF and polynomial.

# LinearSVR
# Module: sklearn.svm
# Linear Support Vector Regression designed for
# large datasets and high-dimensional feature spaces.

# NuSVR
# Module: sklearn.svm
# SVR variant that uses the nu parameter to control
# aspects of support vectors and training errors.


# 6. NEAREST NEIGHBOR REGRESSION
# Predict numerical values based on nearby observations
# in the feature space.

# KNeighborsRegressor
# Module: sklearn.neighbors
# Predicts values using the average or weighted average
# of the K nearest training observations.

# RadiusNeighborsRegressor
# Module: sklearn.neighbors
# Predicts values using observations located within
# a predefined distance radius.


# 7. BAYESIAN REGRESSION MODELS
# Use Bayesian statistics to estimate regression parameters
# and represent uncertainty in predictions.

# ARDRegression
# Module: sklearn.linear_model
# Bayesian regression that applies individual regularization
# to each feature coefficient.

# BayesianRidge
# Module: sklearn.linear_model
# Estimates regression coefficients using Bayesian
# inference with automatic regularization.


# 8. NEURAL NETWORK REGRESSION
# Learn complex nonlinear relationships through layers
# of interconnected artificial neurons.

# MLPRegressor
# Module: sklearn.neural_network
# Multi-Layer Perceptron that uses hidden layers and
# nonlinear activation functions to predict continuous values.


# 9. GENERALIZED LINEAR MODELS (GLM)
# Extend linear regression to model different types
# of target distributions and relationships.

# PoissonRegressor
# Module: sklearn.linear_model
# Models non-negative count data using a Poisson
# distribution.

# GammaRegressor
# Module: sklearn.linear_model
# Models strictly positive continuous target values
# using a Gamma distribution.

# TweedieRegressor
# Module: sklearn.linear_model
# Models target variables using Tweedie distributions,
# including different combinations of zero and positive values.





# CREATING FEATURES
import pandas as pd
import numpy as np

sales_df = pd.DataFrame({
    {'TV' : [100, 345, 231, 193, 200],
     'radio' : [30, 100, 54, 62, 89],
     'Sales' : [254, 453, 183, 472, 212]}
})

# Create X from the radio column's values
X = sales_df['radio'].values

# Create y from the sales column's values
y = sales_df['sales'].values

# Reshape X. Numpy array restructures the data with format (rows, cols), where -1 means numpy guess the number of rows/cols needed
X = X.reshape(-1, 1)

# Check the shape of the features and targets
print(X.shape, y.shape)

# EXAMPLE OF LINEARREGRESSION MODEL

# Import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib as plt

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X_train, y_train)

# Make predictions
y_pred = reg.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))

# Create scatter plot
plt.scatter(X, y, color="blue")

# Create line plot
plt.plot(X, y_pred, color="red")
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")

# Display the plot
plt.show()


# REGRESSION PERFORMANCE METRICS

# SOME USEFUL METRICS

# 1. REGRESSION METRICS
# Evaluate how accurately a model predicts continuous values.

# MAE (Mean Absolute Error)
# Function: mean_absolute_error
# Module: sklearn.metrics
# Measures the average absolute difference between actual
# and predicted values. Lower is better.

# MSE (Mean Squared Error)
# Function: mean_squared_error
# Module: sklearn.metrics
# Measures the average squared prediction error.
# Penalizes large errors more heavily. Lower is better.

# RMSE (Root Mean Squared Error)
# Function: root_mean_squared_error
# Module: sklearn.metrics
# Square root of MSE. Expresses prediction error in the
# same units as the target variable. Lower is better.

# R2 (R-squared)
# Function: r2_score
# Module: sklearn.metrics
# Measures the proportion of target variance explained
# by the model. Higher is better. Maximum: 1.

# MAPE (Mean Absolute Percentage Error)
# Function: mean_absolute_percentage_error
# Module: sklearn.metrics
# Measures prediction error as a percentage of actual values.
# Lower is better. Avoid when actual values are zero or near zero.

# Import root_mean_squared_error
from sklearn.metrics import root_mean_squared_error

# Compute R-squared
r_squared = reg.score(X_test, y_test)

# Compute RMSE
rmse = root_mean_squared_error(y_test, y_pred)

# Print the metrics
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))


