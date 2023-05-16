# EE399 Homework 5

### Author : Hana McVicker

EE399: Overview of Machine Learning Method with a Focus on Formulating the Underlying Optimization Problems Required to Solve Engineering and Science Related Problems

## Abstract
This project explores the use of neural networks for predicting future states of the Lorenz equations with variable ρ values. First, a feed- forward neural network is trained to get solutions from time t to t + ∆t for ρ = 10, 28, and 40. The performance of this neural network is  evaluated for future state prediction at ρ = 17 and ρ = 35. The feed-forward neural network, LSTM, RNN, and Echo State Networks are compared to determine the most effective architecture for forecasting the system's dynamics, offering insights into the use of machine learning techniques in chaos theory prediction.

## Sec. I Introduction and Overview
The field of chaos theory, specifically in the context of the Lorenz equations, presents a unique challenge and opportunity for the application of machine learning techniques, such as neural networks, for predicting future states. The Lorenz equations are a system of three differential equations that demonstrate chaotic dynamics for certain parameter values. The complexity of these equations calls for innovative computational approaches, like neural networks, that can handle non-linearity and unpredictability, common characteristics of such systems. In this assignment, I aim to investigate the predictive potential of various neural network architectures focusing on the Lorenz equations. This network is trained to predict subsequent states of the Lorenz equations for specific ρ values (10, 28, and 40), and its performance is then evaluated for ρ values of 17 and 35. I also compare the feed-forward network with LSTM, RNN, and Echo State Networks to see which architecture has the most accurate forecasting for the dynamics of the Lorenz system.
## Sec. II Theoretical Background
Feed-Forward Neural Networks
Feed-forward neural networks (FNNs) are a fundamental type of artificial neural network widely used for various tasks in machine learning and pattern recognition. The architecture of an FNN consists of an input layer, one or more hidden layers, and an output layer. Information flows in a unidirectional manner, from the input layer through the hidden layers to the output layer, without any feedback connections.

Long Short-Term Memory Networks (LSTM)
LSTM networks are a type of recurrent neural network (RNN) architecture specifically designed to address the issue of capturing long-term dependencies in sequential data. Unlike FNNs, LSTM networks introduce recurrent connections that enable information to be carried across multiple time steps. This recurrent nature makes LSTMs effective in processing sequential data, such as natural language sentences, speech signals, and time series data.

Recurrent Neural Networks (RNN)
RNNs are a class of neural networks that are designed to handle sequential and temporal data. Unlike FNNs, RNNs have feedback connections that enable information to be carried across different time steps, allowing them to process sequences of arbitrary length. RNNs maintain an internal hidden state that evolves as new inputs are processed, capturing dependencies and context from previous time steps.The basic structure of an RNN consists of a single recurrent layer. Each neuron in the layer receives inputs from both the previous time step and the current time step. The activation of the neuron is determined by a combination of the current input and the previous hidden state. This recurrent connectivity allows RNNs to model temporal dependencies and make predictions based on past information.

Echo State Networks (ESNs)
Echo State Networks (ESNs) are a type of recurrent neural network that emphasizes simplicity and computational efficiency. ESNs are composed of a large reservoir of recurrently connected neurons, often referred to as the echo state, which has random fixed weights. The input data is fed into the reservoir, and the reservoir dynamics transform the input into a higher-dimensional representation. The transformed data is then mapped to the desired output using a trainable readout layer. The key property of ESNs is that only the weights of the readout layer are trained, while the reservoir weights remain fixed. This property simplifies the training process and reduces the computational burden. ESNs have been successfully applied to various tasks, including time series prediction.

## Sec. III Algorithm Implementation and Development
                       
```
```

## Sec. IV Computational Results

## Sec. V Summary and Conclusions
