# EE399 Homework 6

### Author : Hana McVicker

EE399: Overview of Machine Learning Method with a Focus on Formulating the Underlying Optimization Problems Required to Solve Engineering and Science Related Problems

## Abstract

In this assignment, I utilize a Long Short-Term Memory (LSTM) decoder to analyze sea-surface temperature data. The approach involves downloading the provided code and data, then training the model and illustrating the results. I perform a detailed performance analysis, focusing on the influence of the time lag variable, the effect of added Gaussian noise to the data, and the impact of different numbers of sensors. These results offer insights into the model's robustness under various conditions, assisting in determining the optimal configuration for reliable and precise temperature forecasts.

## Sec. I Introduction and Overview
The analysis of sea-surface temperature data is an increasingly crucial area of study in environmental science. For this assignment, I will be using a Long Short-Term Memory (LSTM) decoder to parse and learn from such data. This assignment is divided into a sequence of interconnected steps. Initially, I will download the necessary code and data, establishing the foundational resources for my analysis. Following this, I will engage in training the LSTM model and plotting the results, a process that should yield an initial set of valuable insights. I will then examine the model's response to various conditions, beginning with an investigation into the effects of the time lag variable. Then, I will introduce Gaussian noise into the data and examine the model's resilience to such disturbances. Finally, I will manipulate the number of sensors and analyze how this variable influences the performance of the model. Collectively, these stages will offer a comprehensive understanding of the LSTM model's behavior, and will guide the identification of an optimal configuration for accurate and reliable sea-surface temperature predictions.

## Sec. II Theoretical Background
Sea-surface temperatures (SSTs) greatly influence global climate patterns and weather phenomena. As such, precise predictions of SSTs are essential for numerous applications, including climate modeling, weather forecasting, and environmental policy-making.      Predicting SSTs is a complex task due to the numerous interconnected physical processes at play. However, the emergence of machine learning, specifically deep learning, has introduced innovative methods to tackle such challenges. One such innovation is the Long Short-Term Memory (LSTM) network, a type of recurrent neural network (RNN) known for its proficiency with sequential data. LSTMs are particularly suited to handle time-series data, like SSTs, due to their design, which allows them to retain information over prolonged periods. This trait is vital for capturing the temporal dynamics inherent in SSTs. The LSTM's specialized gating mechanisms – input, forget, and output gates – manage the flow of information, mitigating issues like gradient vanishing or exploding that often hinder traditional RNNs. This capability enables LSTMs to model complex, long-term dependencies in sequential data.
In this assignment, I will evaluate the LSTM model's performance under different conditions, focusing on the time lag, noise, and the number of sensors. The time lag represents the interval between inputs and their corresponding outputs in sequence prediction. It's anticipated that the LSTM model's performance may vary with changes in the time lag due to the temporal characteristics of SST data. Introducing noise in the form of Gaussian noise represents potential random disturbances that may be found in real-world data. Examining the robustness of the LSTM model against such noise is vital for assessing its practical utility. Lastly, adjusting the number of sensors corresponds to modifying the amount of data sources or input features for the model. Evaluating performance as a function of the number of sensors will provide insights into how the model handles multidimensionality and data diversity.

## Sec. III Algorithm Implementation and Development

### The Code below until said otherwise is given by Professor in this github repository: https://github.com/Jan-Williams/pyshred
To start, 3 sensor locations are randomly selected and set the trajectory length (lags) to 52, corresponding to one year of measurements:
```
num_sensors = 3 
lags = 52
load_X = load_data('SST')
n = load_X.shape[0]
m = load_X.shape[1]
sensor_locations = np.random.choice(m, size=num_sensors, replace=False)
```
In this code, initial parameters are set where variables num_sensors and lags are set to 3 and 52, respectively, establishing the number of sensors to be sampled from the data and the time lag for the analysis. The dimensions of this matrix are extracted using the shape attribute, with the number of rows (n) and the number of columns (m) being separately stored. Finally, a set of num_sensors sensors is randomly selected from the total available sensors using the np.random.choice function, without replacement. The indices of these selected sensors are stored in the variable sensor_locations.

We now select indices to divide the data into training, validation, and test sets:
```
train_indices = np.random.choice(n - lags, size=1000, replace=False)
mask = np.ones(n - lags)
mask[train_indices] = 0
valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]]
valid_indices = valid_test_indices[::2]
test_indices = valid_test_indices[1::2]
```
First, 1000 unique random indices are selected from the range [0, n - lags), where n is the number of data instances and lags is the time lag for the analysis, using the np.random.choice function. These indices correspond to the instances that will be included in the training set. The selection is stored in the variable train_indices. Next, a binary mask of size n - lags is created, where each entry is initially set to 1. The mask is then updated such that each entry corresponding to a training index is set to 0. This mask effectively distinguishes between instances chosen for training (0) and those not chosen for training (1). The code then proceeds to identify all indices that are not part of the training set (mask!=0), which are stored in valid_test_indices. These indices will be used to create the validation and test sets. Finally, the validation and test sets are formed by alternatively picking indices from valid_test_indices. The validation set (valid_indices) comprises every second index starting from the first one, while the test set (test_indices) includes every second index starting from the second one. This approach ensures that the validation and test sets are disjoint and cover all non-training instances.
sklearn's MinMaxScaler is used to preprocess the data for training and we generate input/output pairs for the training, validation, and test sets. 
```
sc = MinMaxScaler()
sc = sc.fit(load_X[train_indices])
transformed_X = sc.transform(load_X)
```

An instance of the MinMaxScaler is created, and the scaler is fitted to the training data load_X[train_indices] using the fit() method. This step calculates the minimum and maximum values of the training data, which will be used for scaling.
```
### Generate input sequences to a SHRED model
all_data_in = np.zeros((n - lags, lags, num_sensors))
for i in range(len(all_data_in)):
    all_data_in[i] = transformed_X[i:i+lags, sensor_locations]
```
An array all_data_in is initialized with zeros, with dimensions (n - lags, lags, num_sensors). This array will store the input sequences for the SHRED model. The subsequent loop iterates over each index in all_data_in. For each index i, the previous lags time steps of the transformed data transformed_X[i:i+lags, sensor_locations] are assigned to the i-th entry of all_data_in. This process generates the input sequences for the SHRED model.

```
### Generate training validation and test datasets both for reconstruction of states and forecasting sensors
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)
```
Generate training, validation, and test datasets: The code determines whether a GPU (CUDA) is available and assigns the device variable accordingly.
The train_data_in tensor is created as a PyTorch tensor from all_data_in by selecting the appropriate indices for the training set.
The valid_data_in tensor is created as a PyTorch tensor from all_data_in by selecting the appropriate indices for the validation set.
The test_data_in tensor is created as a PyTorch tensor from all_data_in by selecting the appropriate indices for the test set.\

```
### -1 to have output be at the same time as final sensor measurements
train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)
train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
```
The first three lines of code contain indices train_indices + lags - 1, which represent the time steps in transformed_X that correspond to the output or target values for the training set. By adding lags - 1 to train_indices, indices are shifted to select the time steps that align with the final sensor measurements. The torch.tensor() function is used to create the tensor from the selected subset of transformed_X. The dtype=torch.float32 argument specifies that the tensor should have a floating-point data type. The .to(device) method is called on the tensor to move it to the specified device, which can be either 'cuda' (if a GPU is available) or 'cpu'. The lines valid_data_out and test_data_out create PyTorch tensors for the validation and test datasets using the corresponding indices (valid_indices and test_indices) and the same process. The  last three lines of code create dataset objects (train_dataset, valid_dataset, test_dataset) that combine the input sequences with their corresponding target or output values. These datasets will be used for training, validation, and testing the SHRED model, respectively.

We train the model using the training and validation datasets.
```
shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=1000, lr=1e-3, verbose=True, patience=5)
```
In this code, we first instantiate the SHRED model by creating an instance of the SHRED class with specific parameters such as the number of sensors (num_sensors), the number of hidden units (hidden_size), the number of hidden layers (hidden_layers), and regularization terms (l1 and l2). The model is then moved to the designated device, either GPU (cuda) or CPU (cpu), using the .to(device) method.
Next, we initiate the training process by calling the fit() function. This function takes the SHRED model (shred), the training dataset (train_dataset), and the validation dataset (valid_dataset) as inputs. Additional hyperparameters are specified, including the batch size (batch_size), the number of epochs (num_epochs), the learning rate (lr), the verbosity level (verbose), and the patience value for early stopping (patience). During training, the model iteratively updates its parameters to minimize the training loss and improve its performance on the validation set. The fit() function returns a list of validation errors or losses at each epoch, which can be used for further analysis or visualization. 

Finally, we generate reconstructions from the test set and print mean square error compared to the ground truth.
```
test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
print(np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth))
```
In this code, we evaluate the performance of the trained SHRED model on the test dataset and calculate a metric to quantify the reconstruction accuracy. First, we apply the trained SHRED model (shred) to the input data of the test dataset (test_dataset.X). The model's .detach().cpu().numpy() method is used to detach the output tensor from the computational graph, move it to the CPU, and convert it to a NumPy array. The resulting tensor represents the reconstructed output values for the input sequences in the test dataset.

To compare the reconstructed values with the ground truth, we use the sc.inverse_transform() method to invert the scaling transformation performed during the preprocessing stage. This step restores the values back to their original scale. Similarly, we apply sc.inverse_transform() to the ground truth target values of the test dataset (test_dataset.Y). The resulting arrays test_recons and test_ground_truth now contain the reconstructed and original (ground truth) values, respectively, in their original scales. Next, we calculate the relative reconstruction error by computing the norm (magnitude) of the element-wise difference between test_recons and test_ground_truth, divided by the norm of test_ground_truth. This measure assesses the overall difference between the reconstructed values and the ground truth values. Finally, the relative reconstruction error is printed using print().

The results are then plotted and discussed in the results section. 

### For the next part of the assignment, an analysis is done on the performance as a function of the time lag variable, as a function of noise (adding Gaussian noise to the data), and as a function of the number of sensors. 
### The Code below is written by me, but utilizes code from the previous part

Because the code is very similar and almost the same except for the variables being used, I will only explain the implementation of the first performance as a function of the time lag variable. As you can see in the code posted, the steps are almost exactly the same. 

I start by defining a list of different lag values to test: 
```
lags2 = [4, 8, 12, 24, 52]
```

I then initialize an empty list to store the results:
```
results = []
```
I then loop over the list of lag values:
```
for lag in lags2:
    # Prepare the input/output pairs for the given lag
    all_data_in = np.zeros((n - lag, lag, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = transformed_X[i:i+lag, sensor_locations]
 ```
 In this code that is being forlooped, an array all_data_in is initialized with zeros. It has dimensions (n - lag, lag, num_sensors), where n represents the total number of data points, lag is the number of previous time steps used as input for prediction, and num_sensors is the number of sensors or features being considered. A loop is then initiated to iterate over each index in all_data_in. For each index i, the code assigns the previous lag time steps of the transformed data transformed_X[i:i+lag, sensor_locations] to the i-th entry of all_data_in. This step creates a sliding window of size lag to extract the input sequence of sensor measurements at each time step. By the end of the loop, all_data_in contains all the input sequences, where each sequence consists of lag time steps and includes the sensor measurements from the specified sensor_locations.
 
 ```
    # Generate training validation and test datasets both for reconstruction of states and forecasting sensors
    # The following lines of code remain the same, only replace 'lags' with 'lag'
    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
    test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

    train_data_out = torch.tensor(transformed_X[train_indices + lag - 1], dtype=torch.float32).to(device)
    valid_data_out = torch.tensor(transformed_X[valid_indices + lag - 1], dtype=torch.float32).to(device)
    test_data_out = torch.tensor(transformed_X[test_indices + lag - 1], dtype=torch.float32).to(device)

    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)

    # Train the model
    shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=50, lr=1e-3, verbose=True, patience=5)

    test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
    print(np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth))

    # Append the result to the results list
    results.append(np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth))

# Convert the results to a numpy array for easier handling
results = np.array(results)
```
This code generates a training validation and test datasets both for reconstruction of states and forecasting sensors. It then trains the model  and appends the results to the results list that was initialized previously. The results array is used to plot the data and show the results. Because this code is from the previous task and was already explained, I will not be explaining the implementation since it can be reread from above.

## Sec. IV Computational Results
 

## Sec. V Summary and Conclusions
