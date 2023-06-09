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
### Task I
For Task I, the data from Homework One is reconsidered:                        
```
X=np.arange(0,31)
Y=np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
```
Using the dataset above, tasks (i - iv) are done
### (i) Fit the data to a three layer feed forward neural network
The  neural network architecture is defined below:
```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # 1 input feature, 10 neurons in the first hidden layer
        self.fc2 = nn.Linear(10, 10)  # 10 neurons in, 10 neurons out in the second hidden layer 
        self.fc3 = nn.Linear(10, 1)  # 10 neurons in, 1 output feature

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # relu activation function
        x = torch.relu(self.fc2(x))  # relu activation function
        x = self.fc3(x)
        return x
  ```     
- ```class Net(nn.Module)``` defines the class ```Net``` that inherits from the ```PyTorch nn.Module``` class.

- ```def __init__(self)``` defines a constructor method for the Net class.

- ```super(Net, self).__init__()``` calls the constructor method of the parent class (```nn.Module```).

 ```
 self.fc1 = nn.Linear(1, 10)
 self.fc2 = nn.Linear(10, 10)
 self.fc2 = nn.Linear(10, 10)
 ```
 - creates a linear layer with 1/10/10 input feature and 10/10/1 output neurons, and assigns it to an instance variable named fc1, fc2, and fc3 respectively.

- ```def forward(self, x)``` defines a forward method for the Net class that takes an input tensor ```x``` as an argument.

- ```x = torch.relu(self.fc1(x))``` applies the ReLU activation function to the output of the first fully connected layer ```fc1``` with the input tensor ```x```, and assigns the result to ```x```.

- ```x = torch.relu(self.fc2(x))``` applies the ReLU activation function to the output of the second fully connected layer ```fc2``` with the input tensor ```x```, and assigns the result to ```x```.

- ```x = self.fc3(x)``` applies the third fully connected layer ```fc3``` to the output of the second layer with the input tensor ```x```, and assigns the result to ```x```.

- ```return x ``` returns the output tensor x from the forward method, which represents the predicted output of the neural network.

### (ii) Using the first 20 data points as training data, fit the neural network. Compute the least-square error for each of these over the training points. Then compute the least square error of these models on the test data which are the remaining 10 data points.

Prepare the dataset by reshaping the data to be 2D, as expected by the neural network: 
```
X = np.arange(0,31).reshape(-1, 1)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53]).reshape(-1, 1)
```
In the first line, I use the ```np.arange()``` function to create an array of numbers ranging from 0 to 30 (inclusive), with a step size of 1. This array is then reshaped using the ```.reshape()``` method to be a 2D array with a single column. The resulting array X contains 31 rows and 1 column, with each row representing a single input value. The second line creates an array Y that contains the corresponding output values for each input value in X. These values are manually specified as an array of integers, and are also reshaped using the ```.reshape()``` method to be a 2D array with a single column. The resulting array Y contains 31 rows and 1 column, with each row representing a single output value that corresponds to the input value at the same index in X.

Split the dataset into training and test sets: 
```
X_train = torch.Tensor(X[:20])
Y_train = torch.Tensor(Y[:20])
X_test = torch.Tensor(X[20:])
Y_test = torch.Tensor(Y[20:])
```
This code creates four PyTorch Tensor objects ```X_train, Y_train, X_test```, and ```Y_test``` that represent the training and testing data for the model.

The first line and second line uses NumPy slicing syntax to extract the first 20 rows of a ```X``` and ```Y```, and then creates a new PyTorch tensor ```X_train ``` and ```Y_train``` from this subset of data using the ```torch.Tensor()``` function. The resulting tensor ```X_train``` and ```Y_train``` contains the first 20 rows of and ```X``` and and ```Y```, respectively.
The third and fourth line also uses slicing to extract the remaining rows of ```X``` and ```Y``` (rows 20 through 30), and creates a new PyTorch tensor ```X_test and Y_test```. ```X_test and Y_test``` contains the remaining 11 rows of ```X``` and ```Y``` respectively.

Create TensorDatasets:
```
train_dataset = TensorDataset(X_train,Y_train)
test_dataset = TensorDataset(X_test,Y_test)
```
This code creates two new TensorDataset objects ```train_dataset``` and ```test_dataset``` by passing the ```X_train/ X_test``` and ```Y_train/Y_test``` tensors as arguments to the constructors. The resulting ```train_dataset``` and ```test_dataset``` objects contains a set of input-output pairs that can be used for training/testing. Each input-output pair consists of an input tensor and a corresponding output tensor : ```(X_train,Y_train)```, ```(X_test,Y_test)```.

Create data loaders:
```
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=11, shuffle=False)
```
The DataLoader objects are used to load the data in batches during training and testing of a machine learning model.
The first argument to each DataLoader constructor is the corresponding TensorDataset object, ```train_dataset / test_dataset```. 
```batch_size``` specifies the number of samples to include in each batch, where the size is set to 10 and 11, meaning that there will be 10 input-output pairs for the ```train_loader``` and each batch will contain all 11 samples in the ```test_loader```. The third argument, shuffle, determines whether the order of the samples should be shuffled for each epoch. In the case of ```train_loader```, shuffle is set to ```True```, which means that the order of the samples will be randomly shuffled for each epoch. This helps to prevent the model from overfitting to the order of the samples in the training set. In the case of ```test_loader```, shuffle is set to ```False```, which means that the order of the samples will be kept fixed during evaluation. This ensures that the model is tested on the same set of samples each time it is evaluated.

Initialize the network and define the loss function and optimizer:
```
net = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
```
The first line creates an instance of the Net class, which initializes the neural network model and consists of three fully connected layers with 1, 10, and 1 neurons, respectively. The Net class also defines the forward pass of the model, which applies a ReLU activation function to the output of the first two layers and returns the output of the third layer. The second line creates an instance of the mean squared error (MSE) loss function. The ```MSELoss``` function computes the mean squared error between the predicted output and the ground truth label. The third line creates an instance of the stochastic gradient descent (SGD) optimizer. This optimizer is used to update the weights of the neural network during training. The ```net.parameters()``` argument specifies the learnable parameters of the model that should be optimized, which are the weights and biases of the fully connected layers. The ```lr``` argument specifies the learning rate, which controls the step size of the optimizer during weight updates.

Train the network:
```
num_epochs = 100
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 2 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
```
In this code, I am training the neural network model. I specify the number of epochs for training to be 100, meaning that the model will iterate through the entire training data set 100 times. I then use a nested for loop to iterate through each batch of data in the ```train_loader``` DataLoader object. The outer loop iterates through the epochs, while the inner loop iterates through each batch. At each iteration, the following steps are done:
- Gradients are set to zero using the ```zero_grad()``` method of the optimizer object. This is necessary to prevent gradient accumulation from previous iterations.
- Feed the batch of inputs to the neural network model using the ```net(inputs)``` method, which produces the predicted outputs.
- Compute the loss between the predicted outputs and the actual labels using the MSE loss function that was defined previously.
- Compute the gradients of the loss with respect to the model parameters using the ```backward()``` method of the loss object.
- Update the weights of the model using the ```step()``` method of the optimizer object based on the computed gradients.
- Print the loss every two steps using the ```print()``` function to monitor the progress of the training process.
By the end of the 100 epochs, the neural network model will have been trained on the training data set to minimize the loss between the predicted and true values

Test the network: 
```
with torch.no_grad():
    total_loss = 0
    for inputs, labels in test_loader:
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
    print('Average loss on the test data: {:.4f}'.format(total_loss / len(test_loader)))
```
First, I used the ```torch.no_grad()``` disable gradient calculations. This is because during testing, we are only interested in evaluating the performance of the model, not updating its parameters. Then, I iterated through each batch of inputs and labels in the ```test_loader```  object. For each batch, I feed the inputs to the neural network model and compute the predicted outputs using the ```net(inputs)``` method.
I then compute the loss between the predicted outputs and the true labels using the same MSE loss function that was used during training. The total loss across all batches is accumulated using the ```total_loss``` variable. Finally, I calculate and print the average loss on the test data set by dividing the total loss by the number of batches in the test data set using the ```len(test_loader)``` function.

### (iii) Repeat (ii) but use the first 10 and last 10 data points as training data. Then fit the model to the test data (which are the 10 held out middle data points). Compare these results to (ii)

For this question the process was exactly the same as the previous section. The only difference was the dataset used, where now I needed to use different training and testing data within the dataset. The new training and testing datasets are shown below:
```
X_train = torch.Tensor(np.concatenate([X[:10], X[-10:]], axis=0))
Y_train = torch.Tensor(np.concatenate([Y[:10], Y[-10:]], axis=0))
X_test = torch.Tensor(X[10:-10])
Y_test = torch.Tensor(Y[10:-10])
```
To split the dataset into training and test sets, and get the first 10 and last 10 data points as my training data and the remaining as test data, I had to concatenate the arrays ```X``` and ```Y``` using the ```np.concatenate()``` function. The ```axis=0``` parameter ensured that the arrays were concatenated vertically.

The results are compared to (ii) in the results section. 

### (iv) Compare the models fit in homework one to the neural networks in (ii) and (iii)

The results are compared in the results section. 

### Task II
For Task II, I train a feedforward neural network on the MNIST data set and performing the analysis on the following questions below. To start, I began by loading in the first 10,000 samples and scaled the data to [0, 1]. 
```
mnist = fetch_openml('mnist_784')
X = mnist.data[:10000] / 255.0  # use the first 1000 samples and Scale the data to [0, 1]
Y = mnist.target[:10000] # use the first 1000 samples
```
### (i) Compute the first 20 PCA modes of the digit images.
Compute the first 20 PCA modes
```
pca = PCA(n_components=20)
pca.fit(X)

first_20_pca_modes = pca.components_

fig, axes = plt.subplots(4, 5, figsize=(10, 10))
axes = axes.ravel()

for i, mode in enumerate(first_20_pca_modes):
    axes[i].imshow(mode.reshape(28, 28), cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(f'PCA Mode {i+1}')

plt.tight_layout()
plt.show()
```
In this code, I am performing PCA on a dataset ```X``` which contains images on numbers. PCA is a technique for reducing the dimensionality of a dataset while retaining most of its variability. I am setting n_components to 20, which means that I want to retain the top 20 principal components. After fitting the PCA model to the dataset, I extract the first 20 principal components using the ```components_``` attribute. These components are stored in ```first_20_pca_modes```. To visualize the principal components, I create a subplot of 4 rows and 5 columns using ```plt.subplots```. I then iterate over the first 20 principal components and use ```imshow()``` to display them as images using a grayscale color map. I set the title of each subplot to ```PCA Mode i```, where i is the index of the principal component. Finally, I use ```plt.tight_layout()``` to improve the spacing between subplots and plt.show() to display the plot. This code allows me to visualize the top 20 principal components of the dataset, which can help gain insight into the underlying structure of the data and identify patterns or correlations between variables.

### (ii) Build a feed-forward neural network to classify the digits for the MNIST data set. Compare the results of the neural network against LSTM, SVM (support vector machines) and decision tree classifiers.

### Part 1 of (ii)- Build a feed-forward neural network
Define the neural network architecture:
```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784) # flatten the input image                         
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
In this code, I define a neural network architecture to classify images from the MNIST dataset, which contains handwritten digits. The architecture consists of three fully connected layers, each with a different number of nodes. The first layer has 784 nodes, which corresponds to the number of pixels in each image. The output layer has 10 nodes, corresponding to the 10 possible classes (digits 0-9). The network uses ReLU activation function after each layer, except the output layer.

Load the MNIST dataset and apply transformations:
```
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
```
I load the MNIST dataset and apply transformations, such as converting images to tensors. 

Create data loaders:
```
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
```
 I create data loaders to batch and shuffle the data during training and testing.

Initialize the network and define the loss function and optimizer
```
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
```
I initialize the network and define the loss function and optimizer. In this case, I use the Cross-Entropy Loss function, which is suitable for multi-class classification problems, and the stochastic gradient descent optimizer. 

Train the network:
```
num_epochs = 10
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
```
I train the network by iterating over the data batches for a specified number of epochs, computing the loss for each batch, backpropagating the error, and updating the weights using the optimizer. I print the loss every 100 steps to monitor the training progress. 

Test the network:
```
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
```
 After training, I test the network on the test data to measure its accuracy. I iterate over the test data batches, compute the outputs, compare them to the labels, and calculate the number of correct predictions. Finally, I print the accuracy of the network on the 10000 test images.

### Part 2 of (ii)- Compare the results of the neural network against LSTM, SVM (support vector machines) and decision tree classifiers

#### SVM and Decision Tree Classifier
This question was completed in the previous homework (Homework 3), so the code is exactly the same. The code implementation and description can be seen there, on task 9.

#### LSTM
In this code, I am creating a Long Short-Term Memory (LSTM) neural network to perform image classification on the MNIST dataset.

Check if CUDA is available and set PyTorch to use GPU or CPU:
```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
First, I check if CUDA is available to enable GPU acceleration.

Define the LSTM network architecture:
```
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
```
Then, I define the LSTM network architecture using PyTorch. The LSTM network has an input size of 28, a hidden size of 128, 2 layers, and 10 output classes.

Load the MNIST dataset and apply transformations:
```
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
```
Create data loaders:
```
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
```
Next, I load the MNIST dataset and apply transformations, and create data loaders for the training and testing data.

Initialize the LSTM network, define the loss function and optimizer:
```
model = LSTMNet(input_size=28, hidden_size=128, num_layers=2, num_classes=10)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
```
 Then, I initialize the LSTM network, define the loss function as cross-entropy loss, and choose an optimizer as stochastic gradient descent with a learning rate of 0.001.

Train the LSTM network:
```
num_epochs = 10
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28, 28).to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
```
To train the LSTM network, I iterate over the training data for 10 epochs, with a batch size of 100, and calculate the loss, backpropagate the gradients, and update the model's weights. During training, I print out the current epoch, the current step, and the loss.

Test the LSTM network:
```
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28, 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print('Accuracy of the LSTM network on images: {} %'.format(100 * correct / total))
```
After training, I test the LSTM network on the testing data and calculate its accuracy. I iterate over the testing data with a batch size of 100, calculate the predicted labels, compare them to the ground truth labels, and calculate the percentage of correctly classified images. Finally, I print out the accuracy of the LSTM network on the testing data.

## Sec. IV Computational Results
### Task I
#### (i)
The data was fit to a neural network that was used in the following tasks below
#### (ii) Using the first 20 data points as training data, fit the neural network. Compute the least-square error for each of these over the training points. Then compute the least square error of these models on the test data which are the remaining 10 data points.
The least - square error for each of the training points and the least square error of these models on the test data are shown below:

<img width="338" alt="image" src="https://user-images.githubusercontent.com/72291173/237007171-adfdb615-2570-4d19-baca-ae6103ea0c73.png">

The average loss on the test data set provides an estimate of the model's performance on new, unseen data. The average loss is high, so it may be overfitting to the training dataset and not generalizing well to new data. I also may not have enough data for the neural network, so that is why the loss/error is so high. The training loss/error as seen above the testing loss/error is also very high.

#### (iii) Repeating (ii) with different data points
Results are shown below:

<img width="342" alt="image" src="https://user-images.githubusercontent.com/72291173/237007977-663766d5-0cb2-46c3-b538-0976cab28a6a.png">

In these results, we can see that the training loss in both part (ii) and part (iii) are pretty similar, as the numbers are pretty large throughout the training. However, the loss was a bit lower than part (ii), but still pretty high at around 13.0871. Again, this could be because there is not enough data to train the neural network.
#### (iv) Compare the Models in Homework One to the neural networks in (ii) and (iii)
In homework one from Question IV, the least squares error for the data is shown below:

![IMG_9CFB9E32C5D3-1](https://user-images.githubusercontent.com/72291173/237009186-96b5fac3-29ca-4632-8705-51e94a78b685.jpeg)

Here, we can see that the training data and test data from Homework One (other than the Polynomial Error for Test) are significantly lower than the neural networks created in this homework. This shows that neural networks are not the best for smaller datasets.

### Task II
#### (i) Compute the first 20 PCA modes of the Digit Images
The results are shown below:

<img width="990" alt="image" src="https://user-images.githubusercontent.com/72291173/237009913-cc5c5526-e9b5-4008-a091-919b65866482.png">

![image](https://user-images.githubusercontent.com/72291173/237009984-1029449d-9774-4823-a541-73d8bb120954.png)

#### (ii) Compare the Results of the Neural Network against LSTM, SVM, and Decision Tree Classifiers
Neural Network Results:

<img width="622" alt="image" src="https://user-images.githubusercontent.com/72291173/237010952-0f2e9d35-6498-4e24-bc2d-d2050cbb1d36.png">

LSTM Results:

<img width="372" alt="image" src="https://user-images.githubusercontent.com/72291173/237010679-ac6adef9-a8d7-463b-9f21-22c38821073d.png">

SVM and Decision Tree Classifier Result:

<img width="462" alt="image" src="https://user-images.githubusercontent.com/72291173/237010569-90537352-788a-49fd-b64d-6d7c77468c3f.png">

As we can see in the results, the SVM performed the best, with the neural network behind only about 4%. The neural network has a lot higher accuracy rate than before, which could be because there is a lot more data than the previous task data. The decision tree classifier was the third accurate, but the LSTM was much worse, and being the lowest at 11.46%. 
## Sec. V Summary and Conclusions
For the first task of the assignment, I had to look at the data from the previous homework, which consisted of an array of X and Y values. I fitted this data to a three-layer feedforward neural network and used the first 20 data points as training data to compute the least-square error for each model over the training points. Then, I computed the least square error of these models on the remaining 10 data points. I repeated this process with the first 10 and last 10 data points as training data and the remaining 10 data points as test data. I compared the results of these two steps and compared the models fit in homework one to the neural networks in both steps. For the second task, I trained a feedforward neural network on the MNIST dataset. I computed the first 20 PCA modes of the digit images and built a feedforward neural network to classify the digits. I also compared the performance of the neural network against LSTM, SVM, and decision tree classifiers. Overall, this assignment allowed me to gain knowledge of neural networks to fit data and classify images. It also required me to evaluate and select the best model for a given task by comparing the performance of different models.
