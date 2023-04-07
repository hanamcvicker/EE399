# EE399 Homework 1: Model Fitting with Least Square Error

### Author : Hana McVicker

EE399: Overview of Machine Learning Method with a Focus on Formulating the Underlying Optimization Problems Required to Solve Engineering and Science Related Problems

## Abstract

This homework involves fitting a cosine function with linear and constant terms to a dataset using least-squares error. The code finds the minimum error and determines the parameters. The results are then used to create a 2D loss landscape by fixing two of the parameters and sweeping through values of the other two parameters. The minimum errors for line, parabola, and 19th-degree polynomial fits on the training data are also computed, and the models are tested on the remaining 10 data points. The process is repeated using only the first and last 10 data points as training data, and the results are compared to the first method.

## Sec. I. Introduction and Overview

In this assignment, a dataset with 31 points is given, and the focus is on exploring the impact of least-squares error on model fitting. A cosine function is used with linear and constant terms to fit the data while minimizing the error between the model predictions and the actual data.

After finding the minimum error and the optimal values of the parameters, two of the parameters will be fixed, and the other two parameters will be varied to generate a 2D loss (error) landscape. This will help to visualize how the error changes with different parameter values and identify any minima in the loss landscape.

Furthermore, the first 20 data points will be used as training data to fit a line, parabola, and 19th-degree polynomial models, and the least-square error will be calculated for each model over the training points. The models' performance will be evaluated by computing the least square error of these models on the test data (the remaining 10 data points). The same process will be repeated using only the first and last 10 data points as training data, and the results will be compared to the first method.

The objective of this exercise is to understand the impact of least-squares error on model fitting and explore different methods for fitting models to data. This analysis will provide a deeper understanding of selecting and evaluating models for different types of datasets.

## Sec. II. Theoretical Background

Least-squares regression is a common method for fitting mathematical models to data. Given a set of data points, the goal is to find the best-fitting model that describes the relationship between the independent and dependent variables. Least-squares error is a widely used technique for finding the best-fitting line or curve that describes the relationship between the independent and dependent variables in a set of data. This method involves minimizing the sum of the squared differences between the predicted values of the model and the actual data points:

<img width="309" alt="image" src="https://user-images.githubusercontent.com/72291173/230564742-09a7aada-0737-4f44-bbd4-547ce6c085da.png">

The method is based on the principle that the best estimate for the model parameters should be the one that minimizes the sum of the squared differences between the predicted values and the actual data points. In other words, the model parameters that minimize the least-squares error are the ones that best fit the data. The least-squares method involves minimizing the sum of the squared differences between the predicted values of the model and the actual data points.

## Sec. III. Algorithm Implementation and Development

**Given Data :**


