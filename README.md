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

### Generate input sequences to a SHRED model
all_data_in = np.zeros((n - lags, lags, num_sensors))
for i in range(len(all_data_in)):
    all_data_in[i] = transformed_X[i:i+lags, sensor_locations]

### Generate training validation and test datasets both for reconstruction of states and forecasting sensors
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

### -1 to have output be at the same time as final sensor measurements
train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
```


## Sec. IV Computational Results
 

## Sec. V Summary and Conclusions
