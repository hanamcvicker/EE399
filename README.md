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
In the Reservoir class, the ```__init__``` method initializes the reservoir's parameters. It takes ```hidden_dim``` (the dimensionality of the reservoir) and ```connectivity``` (the sparsity of the reservoir connections) as arguments. The weight matrices ```Wx, Wh, and Uh ``` for the reservoir are initialized using the ```sparse_matrix``` method. The activation function ```act``` is set to hyperbolic tangent ```(nn.Tanh())```. The ```sparse_matrix``` method generates a sparse weight matrix with size ```m``` and sparsity probability ```p```. It first creates a random matrix ```S``` and then applies a binary mask obtained from a Bernoulli distribution to the random matrix. Finally, the sparse matrix is returned.

The ```forward``` method performs the forward pass of the reservoir. It takes the input ```x``` and the reservoir state ```h``` as arguments. The method applies the reservoir dynamics by multiplying the input and reservoir state with the corresponding weight matrices using sparse matrix multiplication ```(torch.sparse.mm)```. The results are passed through the activation function ```act```. The output ```y``` is computed similarly. Finally, the output ```y``` and the updated state ```h``` are returned.

In the EchoState class, the ```__init__ ``` method initializes the ESN's parameters. It takes ```in_dim ```(the dimensionality of the input), ```out_dim``` (the dimensionality of the output), ```reservoir_dim``` (the dimensionality of the reservoir), and ```connectivity``` (the sparsity of the reservoir connections) as arguments. It initializes the ```input-to-reservoir``` linear transformation ```input_to_reservoir``` and sets its gradients to be excluded from training. It also initializes the reservoir using the ```Reservoir``` class and sets up the readout layer (```readout```) to map from the reservoir to the output dimension.

The ```forward``` method performs the forward pass of the ESN. It takes the input ```x``` as an argument. The input is first transformed using the ```input_to_reservoir``` linear layer. The initial reservoir state ```h``` is initialized as a tensor of ones. The reservoir dynamics are then applied iteratively for each time step in the input sequence. The input at each time step is processed by the reservoir, and the output of the reservoir is collected. The resulting reservoir outputs are concatenated along the time dimension. Finally, the concatenated reservoir outputs are passed through the readout layer to produce the final output.

These Models are then used for the training and testing. Because the code for each model is very similar, I will explain the implementation of the training and testing code for the FFNN only. 

### Training
This code is not repeated in the other models, as all the training data is the same, so it is done where the FFNN model training is. 
```
# Generate training data
rho_train = [10, 28, 40]
nn_input = np.zeros((0, 3))
nn_output = np.zeros_like(nn_input)

for i, rho in enumerate(rho_train):
        nn_input_rho, nn_output_rho = lorenz(rho)
        nn_input = np.concatenate((nn_input, nn_input_rho))
        nn_output = np.concatenate((nn_output, nn_output_rho))  
 ```
 This code constructs the training dataset by generating input-output pairs for the Lorenz system with different rho values. The resulting arrays, ```nn_input``` and ```nn_output```, hold the input and output data required for training the neural network.
 
 
This code is repeated on the other models, with the same format
 ```
# Create model instance
model = MyModel()
```
The model is defined using the ```MyModel``` class, which represents the feed-forward neural network in this example.
```
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```
The loss function is defined using nn.MSELoss(), the Mean Squared Error loss
The optimizer is defined as stochastic gradient descent (SGD) with a learning rate of 0.01
 ```
# Convert numpy arrays to PyTorch tensors
nn_input_torch = torch.from_numpy(nn_input).float()
nn_output_torch = torch.from_numpy(nn_output).float()
```
This converts the numpy arrays to PyTorch tensors and made into the float data type

```
# Train the model
for epoch in range(30):
    optimizer.zero_grad()
    outputs = model(nn_input_torch)
    loss = criterion(outputs, nn_output_torch)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, loss={loss.item():.4f}")
```
This code is a loop where the model is optimized by minimizing the loss. The optimizer updates the model's parameters based on the computed gradients, gradually improving the model's performance as the epochs progress. The print statement provides periodic updates on the loss value during training, allowing for monitoring and evaluation of the model. 

As shown, all of the models follow similar/ if not the same format:
Creating model instance
Defining loss function and optimizer
Reshaping the input data
Training the model

### Testing
Testing code for the FFNN:
```
# Testing for future state prediction for ρ = 17 and ρ = 35.
test_values = [17, 35]

for rho in test_values:
    ffnn_test_input, ffnn_test_output = lorenz(rho)
    ffnn_test_input = torch.from_numpy(ffnn_test_input).float()
    ffnn_test_output = torch.from_numpy(ffnn_test_output).float()
    ffnn_output_pred = model(ffnn_test_input)
    loss = criterion(ffnn_output_pred, ffnn_test_output)
    print('Loss for rho = ', rho, ': ', loss.item())
```
This code evaluates the performance of the FFNN model for future state prediction on the test data corresponding to different values of rho. It computes the loss between the predicted outputs and the target outputs, providing a measure of how well the model performs for each rho value. 

As previously stated, the models follow the same format as the FFNN shown above. 

## Sec. IV Computational Results
The Results for the Models are shown below:
### FFNN

<img width="182" alt="image" src="https://github.com/hanamcvicker/EE399/assets/72291173/3fb78d35-595d-42a3-8494-e0705d0e2916">

<img width="317" alt="image" src="https://github.com/hanamcvicker/EE399/assets/72291173/4400cd6a-a2a2-4d13-a918-1773e3ffbefc">

### LSTM

<img width="204" alt="image" src="https://github.com/hanamcvicker/EE399/assets/72291173/f7285677-cada-41c6-92a5-cd3768c72d92">

<img width="311" alt="image" src="https://github.com/hanamcvicker/EE399/assets/72291173/ecd6f6d1-3975-4b2d-8162-afac0af06748">

### RNN

<img width="183" alt="image" src="https://github.com/hanamcvicker/EE399/assets/72291173/e8cec56e-9cd6-48ef-9d19-3b0a147b58ba">

<img width="325" alt="image" src="https://github.com/hanamcvicker/EE399/assets/72291173/7922e167-bfb6-4c93-bbd2-2176ec90f5a8">

### ESN

<img width="189" alt="image" src="https://github.com/hanamcvicker/EE399/assets/72291173/c57ac62a-80ec-41c0-a214-f8ed77838502">

<img width="317" alt="image" src="https://github.com/hanamcvicker/EE399/assets/72291173/8a73f1aa-a4c7-41dd-8a58-93bc2bbb7d06">

As seen from the results, the training and loss is lowest in the ESN and RNN, as the losses are all three double digits. The FFNN and LSTM have training and testing losses that are triple digits, which show that those models are not best suited for predicting future states as wel as the ESN and RNN. 

## Sec. V Summary and Conclusions

