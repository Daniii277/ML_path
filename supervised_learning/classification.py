''' 
Classification : Classification is used to predict the label, or category, of an observation. 
                For example, we can predict whether a bank transaction is fraudulent or not. 
                As there are two outcomes here - a fraudulent transaction, or non-fraudulent transaction, 
                this is known as binary classification.
'''

###############    SOME SKLEARN SUPERVISED LEARNING AND CLASSIFICATION MODELS    ####################


# 1. LINEAR CLASSIFIERS
# Linear models that learn decision boundaries to separate
# different classes.

# LogisticRegression
# Module: sklearn.linear_model
# Estimates class probabilities using the logistic function.
# Supports L1, L2 and other regularization options.

# RidgeClassifier
# Module: sklearn.linear_model
# Uses linear regression with L2 regularization to perform
# classification based on decision scores.

# SGDClassifier
# Module: sklearn.linear_model
# Trains linear classifiers using Stochastic Gradient Descent.
# Supports different loss functions such as hinge and log_loss.

# PassiveAggressiveClassifier
# Module: sklearn.linear_model
# Updates the model when predictions are incorrect.
# Particularly useful for large-scale and online learning.

# Perceptron
# Module: sklearn.linear_model
# Implements a classical linear classification algorithm
# that learns weights to separate different classes.


# 2. DISTANCE-BASED AND NEAREST NEIGHBOR CLASSIFIERS
# Classify observations based on their distance and similarity
# to other observations in the training dataset.

# KNeighborsClassifier
# Module: sklearn.neighbors
# Assigns a class based on the majority vote of the K nearest
# neighbors.

# RadiusNeighborsClassifier
# Module: sklearn.neighbors
# Classifies observations using neighbors located within
# a predefined distance radius.

# NearestCentroid
# Module: sklearn.neighbors
# Classifies observations according to the closest class
# centroid.


# 3. DECISION TREE CLASSIFIERS
# Build decision rules by recursively splitting the dataset
# according to feature values.

# DecisionTreeClassifier
# Module: sklearn.tree
# Creates a decision tree using splitting criteria such as
# Gini impurity or entropy.

# ExtraTreeClassifier
# Module: sklearn.tree
# Builds a decision tree using more randomized split
# selection.


# 4. ENSEMBLE CLASSIFIERS
# Combine multiple models to improve predictive performance,
# robustness and generalization.

# RandomForestClassifier
# Module: sklearn.ensemble
# Combines multiple decision trees trained on different
# bootstrap samples and feature subsets.

# ExtraTreesClassifier
# Module: sklearn.ensemble
# Combines randomized decision trees to reduce variance
# and improve generalization.

# BaggingClassifier
# Module: sklearn.ensemble
# Trains multiple models on different bootstrap samples
# and combines their predictions.

# AdaBoostClassifier
# Module: sklearn.ensemble
# Sequentially trains weak learners, giving more importance
# to observations incorrectly classified by previous models.

# GradientBoostingClassifier
# Module: sklearn.ensemble
# Builds sequential models that attempt to correct the
# errors made by previous models.

# HistGradientBoostingClassifier
# Module: sklearn.ensemble
# Optimized gradient boosting implementation that uses
# histogram-based learning for faster training.

# VotingClassifier
# Module: sklearn.ensemble
# Combines predictions from multiple classifiers using
# majority voting or weighted probability averaging.

# StackingClassifier
# Module: sklearn.ensemble
# Combines several classifiers using a final estimator
# (meta-model) that learns how to integrate their predictions.


# 5. NAIVE BAYES CLASSIFIERS
# Apply Bayes' theorem to estimate the probability that
# an observation belongs to a particular class.

# GaussianNB
# Module: sklearn.naive_bayes
# Assumes that features follow a Gaussian distribution
# within each class.

# MultinomialNB
# Module: sklearn.naive_bayes
# Uses a multinomial distribution. Commonly used for
# text classification and word frequency data.

# BernoulliNB
# Module: sklearn.naive_bayes
# Designed for binary-valued features, such as the presence
# or absence of words in documents.

# ComplementNB
# Module: sklearn.naive_bayes
# A Naive Bayes variant particularly suited to imbalanced
# text classification datasets.

# CategoricalNB
# Module: sklearn.naive_bayes
# Designed for categorical features with discrete values.


# 6. SUPPORT VECTOR MACHINES (SVM)
# Find decision boundaries that maximize the margin
# between different classes.

# SVC
# Module: sklearn.svm
# Support Vector Classifier that supports linear and
# nonlinear kernels (linear, polynomial, RBF and sigmoid).

# NuSVC
# Module: sklearn.svm
# SVM variant that uses the nu parameter to control
# the number of support vectors and margin errors.

# LinearSVC
# Module: sklearn.svm
# Linear Support Vector Classifier optimized for large
# datasets and high-dimensional feature spaces.


# 7. DISCRIMINANT ANALYSIS CLASSIFIERS
# Model class distributions to establish decision boundaries.

# LinearDiscriminantAnalysis (LDA)
# Module: sklearn.discriminant_analysis
# Finds linear combinations of features that maximize
# class separation. Assumes shared class covariance.

# QuadraticDiscriminantAnalysis (QDA)
# Module: sklearn.discriminant_analysis
# Uses quadratic decision boundaries and allows each
# class to have its own covariance matrix.


# 8. NEURAL NETWORK CLASSIFIERS
# Learn complex and nonlinear patterns through layers
# of interconnected artificial neurons.

# MLPClassifier
# Module: sklearn.neural_network
# Multi-Layer Perceptron that uses hidden layers and
# nonlinear activation functions to classify observations.


# 9. PROBABILITY CALIBRATION
# Adjust the predicted probabilities of an existing classifier
# to make them better reflect actual class frequencies.

# CalibratedClassifierCV
# Module: sklearn.calibration
# Calibrates classifier probabilities using methods such as
# sigmoid (Platt scaling) or isotonic regression.


#EXAMPLE WITH K-NEAREST NEIGHBORS

# Import the module
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib as plt

#dataFrame could be created by dictionary, csv or numpy array.
churn_df = {'data 1' : [1,2,3,4,5,6],
            'data 2' : [3,6,5,1,3,2],
            'churn' : [9,7,1,8,3,4]}

#Drop the target variable and select the features
X = churn_df.drop("churn", axis=1).values
y = churn_df["churn"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy. Score measures the true positives got by the model divided by total of predictions
print(knn.score(X_test, y_test))

# OVERFITTING AND UNDERFITTING

'''
Overfitting : Occurs when the model learns the training data very well, incluiding noise and irrelevant patterns.
              The model performs accurate with training data and performs poorly with unseen data.
			  
Underfitting : occurs when the model is too simple and doesnt learn patterns correctly. The model performs poorly
                on both training and test data.
'''
# EXAMPLE TESTING UNDERFITTING AND OVERFITTING DEPENDING ON THE NUMBER OF NEIGHBORS DEFINED
# Create neighbors
neighbors = np.arange(1, 13)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
  
	# Set up a KNN Classifier
	knn = KNeighborsClassifier(n_neighbors=neighbor)
  
	# Fit the model
	knn.fit(X_train, y_train)
  
	# Compute accuracy
	train_accuracies[neighbor] = knn.score(X_train, y_train)
	test_accuracies[neighbor] = knn.score(X_test, y_test)
print(neighbors, '\n', train_accuracies, '\n', test_accuracies)

# VISUALIZING RESULTS WITH MATPLOTLIB

# Add a title
plt.title("KNN: Varying Number of Neighbors")

# Plot training accuracies
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")

# Plot test accuracies
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")

plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")

# Display the plot
plt.show()

# CLASSIFICATION PERFORMNACE

# SOME USEFUL METRICS

# 2. CLASSIFICATION METRICS
# Evaluate how accurately a model predicts discrete classes.

# Accuracy
# Function: accuracy_score
# Module: sklearn.metrics
# Measures the proportion of correctly classified observations.

# Precision
# Function: precision_score
# Module: sklearn.metrics
# Measures how many predicted positive observations are
# actually positive. Higher precision means fewer false positives.

# Recall
# Function: recall_score
# Module: sklearn.metrics
# Measures how many actual positive observations are
# correctly identified. Higher recall means fewer false negatives.

# F1-Score
# Function: f1_score
# Module: sklearn.metrics
# Harmonic mean of precision and recall.
# Useful when balancing both metrics.

# ROC-AUC
# Function: roc_auc_score
# Module: sklearn.metrics
# Measures the model's ability to distinguish between classes.
# Higher values indicate better discrimination.

# Confusion Matrix
# Function: confusion_matrix
# Module: sklearn.metrics
# Displays true positives, true negatives, false positives
# and false negatives.
