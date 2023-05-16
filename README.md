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

After defining the lorenz function, I defined the activation functions below. Activation functions are crucial in neural networks as they introduce nonlinearity, enabling the network to learn complex patterns and relationships in the data.
```
def logsig(x):
    return 1 / (1 + torch.exp(-x))

def radbas(x):
    return torch.exp(-torch.pow(x, 2))

def purelin(x):
    return x
```
The logsig function represents the logistic sigmoid activation function. It takes the input x and applies the sigmoid function to map the values to a range between 0 and 1, where the sigmoid function is defined as 1 / (1 + exp(-x)). By using the logsig activation function, the output is transformed into a probability or activation level, where values close to 0 approach 0 and values close to infinity approach 1

The radbas function represents a radial basis function. It takes the input x and applies the exponential of the negative squared value of x to produce a bell-shaped curve. Mathematically, it is represented as exp(-x^2). The radbas activation function is commonly used in tasks such as pattern recognition and approximation. 

The purelin function is a simple identity function. It returns the input x as the output without any transformation. 

After defining the activation functions, I created each network by creating each of the models, and then training each one to advance the solution from t to t + ∆t for ρ = 10, 28 and 40 and then testing each one to see how well it works for future state prediction for ρ = 17 and ρ = 35. 

Because the code is quite similar for the training and testing of each model, I will first go through the implementation of each network. 

### Feed - Forward Neural Network Model
The code for defining the FFNN is shown below:
```
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(in_features=3, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=10)
        self.fc3 = nn.Linear(in_features=10, out_features=3)

    def forward(self, x):
        x = logsig(self.fc1(x))
        x = radbas(self.fc2(x))
        x = purelin(self.fc3(x))
        return x
```
This code represents a three-layer feed-forward neural network. It takes an input x, passes it through the layers sequentially, applies different activation functions at each layer, and produces the final output of the network. In the constructor __init__, the model's architecture is defined. Three fully connected layers (nn.Linear) are created: self.fc1, self.fc2, and self.fc3. These layers specify the connectivity and number of neurons in each layer.

In the forward method, the forward pass of the model is defined. The input x is fed through the layers in a sequential manner. The output of each layer is passed through an activation function before being fed into the next layer. The logsig function is applied to the output of self.fc1, the radbas function is applied to the output of self.fc2, and the purelin function is applied to the output of self.fc3. 

### LSTM and RNN Model
The code for defining the LSTM and RNN is shown below:
```
class LSTM(nn.Module):
    def __init__(self, input_size=3, hidden_layer_size=10, output_size=3):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_layer_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_layer_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out
```
The code for defining the RNN is shown below:
```
class RNN(nn.Module):
    def __init__(self, input_size=3, hidden_layer_size=10, output_size=3):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.rnn = nn.RNN(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_layer_size)
        out, _ = self.rnn(x, h0)
        out = self.linear(out[:, -1, :])
        return out
```
Both constructors takes input size, hidden layer size, and output size as parameters. The hidden layer size represents the number of hidden units or neurons in the model. The models also both have a recurrent layer followed by a linear layer. In the LSTM model, the recurrent layer is defined as ```self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)```. In the RNN model, the recurrent layer is defined as ```self.rnn = nn.RNN(input_size, hidden_layer_size, batch_first=True)```. These recurrent layers implement the LSTM and RNN architectures, respectively. They take the input size, hidden layer size, and a ```batch_first=True argument```, indicating that the input data has the batch dimension as the first dimension.

In the forward method of both models, the forward pass is defined. The input x is processed through the recurrent layer, capturing the temporal dependencies in the data. For the LSTM model, the initial hidden and cell states are initialized as zero tensors. The output of the recurrent layer is then passed through the linear layer (self.linear) to produce the final output. For the RNN model, only the initial hidden state is initialized as zero tensors. The rest of the process is similar, where the output of the recurrent layer is fed into the linear layer to produce the final output.

### Echo State Model
The code for defining the ESN is shown below:
```
class Reservoir(nn.Module):
  def __init__(self, hidden_dim, connectivity):
    super().__init__()
    
    self.Wx = self.sparse_matrix(hidden_dim, connectivity)
    self.Wh = self.sparse_matrix(hidden_dim, connectivity)
    self.Uh = self.sparse_matrix(hidden_dim, connectivity)
    self.act = nn.Tanh()

  def sparse_matrix(self, m, p):
    mask_distribution = torch.distributions.Bernoulli(p)
    S = torch.randn((m, m))
    mask = mask_distribution.sample(S.shape)
    S = (S*mask).to_sparse()
    return S

  def forward(self, x, h):
    h = self.act(torch.sparse.mm(self.Uh, h.T).T +
                 torch.sparse.mm(self.Wh, x.T).T)
    y = self.act(torch.sparse.mm(self.Wx, h.T).T)

    return y, h
     
class EchoState(nn.Module):
  def __init__(self, in_dim, out_dim, reservoir_dim, connectivity):
    super().__init__()

    self.reservoir_dim = reservoir_dim
    self.input_to_reservoir = nn.Linear(in_dim, reservoir_dim)
    self.input_to_reservoir.requires_grad_(False)

    self.reservoir = Reservoir(reservoir_dim, connectivity)
    self.readout = nn.Linear(reservoir_dim, out_dim)
  
  def forward(self, x):
    reservoir_in = self.input_to_reservoir(x)
    h = torch.ones(x.size(0), self.reservoir_dim)
    reservoirs = []
    for i in range(x.size(1)):
      out, h = self.reservoir(reservoir_in[:, i, :], h)
      reservoirs.append(out.unsqueeze(1))
    reservoirs = torch.cat(reservoirs, dim=1)
    outputs = self.readout(reservoirs)
    return outputs
```
## Sec. IV Computational Results

## Sec. V Summary and Conclusions
