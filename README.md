# EE399 Homework 4

### Author : Hana McVicker

EE399: Overview of Machine Learning Method with a Focus on Formulating the Underlying Optimization Problems Required to Solve Engineering and Science Related Problems

## Abstract
In this assignment, the performance of three-layer feed-forward neural networks is analyzed with fitting time series data and also classifies MNIST digit images. Two nueral network models are created and are compared with the results of the previous homework 1. For the MNIST dataset, the first 20 PCA modes are computed and compared with the neural network's classification performance with LSTM networks, Support Vector Machines, and Decision Tree classifiers. The analysis aims to assess feed-forward neural networks' efficacy in diverse applications and against alternative machine learning models.


## Sec. I Introduction and Overview
In this assignment, I revisit the time series data from homework one and investigate the performance of a three-layer feed-forward neural network for data fitting and prediction. The dataset comprises 31 data points, and two different training strategies are explored: (i) using the first 20 data points for training, and (ii) using the first 10 and last 10 data points for training. The least-square error is calculated over the training points and the test data, comparing the neural network models with the models fitted in homework one. Furthermore, a feed-forward neural network is applied to the MNIST dataset for digit classification. Prior to training the neural network,the first 20 PCA modes are computed of the digit images to analyze the primary sources of variance in the dataset. A feed-forward neural network is then constructed for digit classification and compared its performance with other machine learning models, such as Long Short-Term Memory (LSTM) networks, Support Vector Machines (SVM), and Decision Tree classifiers. This comparative analysis aims to assess the efficacy of feed-forward neural networks for both time series and image classification tasks.

## Sec. II Theoretical Background

Feed-forward neural networks (FFNNs) are a class of artificial neural networks characterized by the absence of cycles or loops in their structure, allowing information to flow in one direction, from input to output. They have been widely used for various applications, including time series prediction and image classification. In time series analysis, FFNNs can capture complex patterns and relationships in sequential data, providing a flexible and powerful tool for modeling and forecasting. The MNIST dataset, a collection of handwritten digits, is a classic image classification problem that has served as a benchmark for machine learning algorithms. Principal Component Analysis (PCA) is a dimensionality reduction technique commonly employed to preprocess high-dimensional data such as images, extracting the most relevant features and facilitating more efficient learning. Comparing FFNNs with alternative machine learning models like Long Short-Term Memory networks (LSTMs), Support Vector Machines (SVMs), and Decision Tree classifiers helps evaluate their performance in diverse applications and provides insights into their strengths and weaknesses. 

## Sec. III Algorithm Implementation and Development

```

```


## Sec. IV Computational Results


## Sec. V Summary and Conclusions

