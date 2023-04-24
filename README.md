# EE399 Homework 3

### Author : Hana McVicker

EE399: Overview of Machine Learning Method with a Focus on Formulating the Underlying Optimization Problems Required to Solve Engineering and Science Related Problems

## Abstract

This assignment involves analyzing the MNIST data set using SVD analysis and building a classifier to identify individual digits. The first part of the analysis involves reshaping the digit images into column vectors and performing an SVD analysis to determine the necessary modes for good image reconstruction. The U, Σ, and V matrices are also interpreted. The data is then projected into PCA space and a linear classifier is built to identify and classify two and three selected digits. The difficulty and accuracy of separating different digit pairs using LDA, SVM, and decision trees are quantified and compared. The performance of the classifier on both the training and test sets is discussed throughout the analysis.

## Sec. I Introduction and Overview

The MNIST data set is a well-known dataset of handwritten digits that has been widely used in machine learning research. The goal of this assignment is to analyze the MNIST data set and build a classifier to identify individual digits in the training set. The analysis will involve using Singular Value Decomposition (SVD) to reshape the digit images and perform an SVD analysis to determine the necessary modes for good image reconstruction. The data will then be projected into PCA space and a linear classifier will be built to identify and classify selected digits. The accuracy and difficulty of separating different digit pairs using Linear Discriminant Analysis (LDA), Support Vector Machines (SVM), and decision trees will be quantified and compared. The performance of the classifier on both the training and test sets will be evaluated.

## Sec. II Theoretical Background

The MNIST dataset is a collection of 28x28 grayscale images of handwritten digits ranging from 0 to 9. Each image is represented as a matrix of pixel values, which can be reshaped into a column vector. In this assignment, the MNIST data set will be analyzed using Singular Value Decomposition (SVD), Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), Support Vector Machines (SVM), and Decision Trees. SVD is a factorization method that has been widely used in machine learning. SVD is used to decompose a matrix A into the product of three matrices: A = UΣV^T, where U and V are orthogonal matrices, and Σ is a diagonal matrix of singular values. PCA is a technique that uses SVD to reduce the dimensionality of high-dimensional data while preserving most of the variation in the data. PCA can be used to project the MNIST dataset into a lower-dimensional space, where the variation in the data is captured by a small number of principal components. LDA is a supervised learning technique that can be used to find a linear combination of features that separates different classes in the data. LDA is often used for dimensionality reduction and classification tasks. SVM is a popular classification algorithm that uses a hyperplane to separate data into different classes. Decision trees are a method for classification and regression tasks. A decision tree is a hierarchical structure that is built by recursively splitting the data based on the values of the input features.

## Sec. III Algorithm Implementation and Development


## Sec. IV Computational Results

### Task (a)



## Sec. V Summary and Conclusions



