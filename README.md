# EE399 Homework 5

### Author : Hana McVicker

EE399: Overview of Machine Learning Method with a Focus on Formulating the Underlying Optimization Problems Required to Solve Engineering and Science Related Problems

## Abstract
This project explores the use of neural networks for predicting future states of the Lorenz equations with variable ρ values. First, a feed- forward neural network is trained to get solutions from time t to t + ∆t for ρ = 10, 28, and 40. The performance of this neural network is  evaluated for future state prediction at ρ = 17 and ρ = 35. The feed-forward neural network, LSTM, RNN, and Echo State Networks are compared to determine the most effective architecture for forecasting the system's dynamics, offering insights into the use of machine learning techniques in chaos theory prediction.

## Sec. I Introduction and Overview
The field of chaos theory, specifically in the context of the Lorenz equations, presents a unique challenge and opportunity for the application of machine learning techniques, such as neural networks, for predicting future states. The Lorenz equations are a system of three differential equations that demonstrate chaotic dynamics for certain parameter values. The complexity of these equations calls for innovative computational approaches, like neural networks, that can handle non-linearity and unpredictability, common characteristics of such systems. In this assignment, I aim to investigate the predictive potential of various neural network architectures focusing on the Lorenz equations. This network is trained to predict subsequent states of the Lorenz equations for specific ρ values (10, 28, and 40), and its performance is then evaluated for ρ values of 17 and 35. I also compare the feed-forward network with LSTM, RNN, and Echo State Networks to see which architecture has the most accurate forecasting for the dynamics of the Lorenz system.
## Sec. II Theoretical Background

The Lorenz equations refer to a system of ordinary differential equations that describe the behavior of a simplified model of atmospheric convection: 
```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz
```
x, y, and z represent the state variables of the system, which represent the convection currents in the atmosphere. The variables σ, ρ, and β are parameters that determine the behavior of the system.

Feed-Forward Neural Networks
Feed-forward neural networks (FNNs) are a fundamental type of artificial neural network widely used for various tasks in machine learning and pattern recognition. The architecture of an FNN consists of an input layer, one or more hidden layers, and an output layer. Information flows in a unidirectional manner, from the input layer through the hidden layers to the output layer, without any feedback connections.

Long Short-Term Memory Networks (LSTM)
LSTM networks are a type of recurrent neural network (RNN) architecture specifically designed to address the issue of capturing long-term dependencies in sequential data. Unlike FNNs, LSTM networks introduce recurrent connections that enable information to be carried across multiple time steps. This recurrent nature makes LSTMs effective in processing sequential data, such as natural language sentences, speech signals, and time series data.

Recurrent Neural Networks (RNN)
RNNs are a class of neural networks that are designed to handle sequential and temporal data. Unlike FNNs, RNNs have feedback connections that enable information to be carried across different time steps, allowing them to process sequences of arbitrary length. RNNs maintain an internal hidden state that evolves as new inputs are processed, capturing dependencies and context from previous time steps.The basic structure of an RNN consists of a single recurrent layer. Each neuron in the layer receives inputs from both the previous time step and the current time step. The activation of the neuron is determined by a combination of the current input and the previous hidden state. This recurrent connectivity allows RNNs to model temporal dependencies and make predictions based on past information.

Echo State Networks (ESNs)
Echo State Networks (ESNs) are a type of recurrent neural network that emphasizes simplicity and computational efficiency. ESNs are composed of a large reservoir of recurrently connected neurons, often referred to as the echo state, which has random fixed weights. The input data is fed into the reservoir, and the reservoir dynamics transform the input into a higher-dimensional representation. The transformed data is then mapped to the desired output using a trainable readout layer. The key property of ESNs is that only the weights of the readout layer are trained, while the reservoir weights remain fixed. This property simplifies the training process and reduces the computational burden. ESNs have been successfully applied to various tasks, including time series prediction.

## Sec. III Algorithm Implementation and Development

In this assignment, I was tasked with training a neural network to advance the solution from t to t + ∆t for ρ = 10, 28 and 40 and see how well it works for future state prediction for ρ = 17 and ρ = 35. Then, the results are compared between a feed-forward, LSTM, RNN and Echo State Networks for forecasting the dynamics.

To start, I defined the lorenz function below: 
```
def lorenz(rho):
    # Initialize parameters
    dt = 0.01
    T = 8
    t = np.arange(0,T+dt,dt)
    beta = 8/3
    sigma = 10
```
This code initializes several parameters: The dt variable represents the time step used in the simulation, while T represents the total duration of the simulation. The t array is created using np.arange to define a sequence of time values from 0 to T with increments of dt. The variables beta and sigma are set to specific values used in the Lorenz equations.
```
    # Initialize input and output arrays for rho value
    nn_input = np.zeros((100 * (len(t) - 1), 3))
    nn_output = np.zeros_like(nn_input)
```
This section creates input and output arrays to store the generated data for the given rho value. The arrays are initialized with zeros using np.zeros. The shape of the arrays is determined by multiplying 100 with the length of the time array t minus 1. This is done to accommodate the input-output pairs generated from the Lorenz system.
```
    # Define Lorenz system
    def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):
        x, y, z = x_y_z
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
```
This section defines the lorenz_deriv function, which represents the derivative of the state variables (x, y, and z) of the Lorenz system with respect to time. The equations of the Lorenz system are implemented within this function.
```
    # Solve Lorenz system for rho value
    np.random.seed(123)
    x0 = -15 + 30 * np.random.random((100, 3))
    x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t)
                    for x0_j in x0])
```
In this section, the code solves the Lorenz system for the given rho value. First, a random seed is set to ensure reproducibility of the generated initial conditions. The x0 array is created by randomly generating values between -15 and 15 to represent the initial state of the system for 100 different trajectories.
The integrate.odeint function is then used to numerically integrate the Lorenz system for each set of initial conditions (x0_j) over the time array t. The resulting trajectories are stored in the x_t array.
```
    for j in range(100):
        nn_input[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_t[j,:-1,:]
        nn_output[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_t[j,1:,:]
    
    return nn_input, nn_output
```
This section iterates over the generated trajectories and populates the input and output arrays with the corresponding data points. The loop runs 100 times, corresponding to the 100 trajectories. For each iteration, the input array (nn_input) is assigned a slice of the x_t array from the first data point to the second-to-last data point for the current trajectory. The output array (nn_output) is assigned a slice of the x_t array from the second data point to the last data point for the current trajectory. 

## Sec. IV Computational Results

## Sec. V Summary and Conclusions
