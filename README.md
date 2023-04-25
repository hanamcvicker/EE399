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

### For the first part, the task was to perform an analysis of the MNIST data set by performing the following analysis:

```
mnist = fetch_openml('mnist_784')
X = mnist.data[:10000] / 255.0  # use the first 1000 samples and Scale the data to [0, 1]
Y = mnist.target[:10000] # use the first 1000 samples
```
To start, I first had to load the MNIST data set. I only loaded the first 10,0000 samples so that it would not take too long to run. I then scaled the 
data by dividing by 255. Dividing the MNIST data by 255 scales the pixel values to the range [0, 1], which normalizes the image data. In MNIST, the pixel values represent grayscale intensities ranging from 0 to 255, so scaling them to can make it easier for the algorithm to learn useful patterns in the data. 

### Task 1)
Do an SVD analysis of the digit images. You will need to reshape each image into a column vector
and each column of your data matrix is a different image.
```
X = X.T
U, S, V = svd(X, full_matrices=False)
```
To do this task, I first reshaped the digit images into column vectors and applied an SVD to the data. The first line transposes the matrix X, meaning that the rows of X become columns and the columns become the rows. I transposed the matrix before applying the SVD since the SVD is usually performed on the columns of the data matrix. Then, the SVD was performed, which decomposes X into three matrices, U, S, and V. Next, the SVD is performed on the transposed matrix X using the svd() function. The SVD decomposes X into three matrices: U, S, and V. U is a matrix containing the left singular vectors of X, S is a diagonal matrix containing the singular values of X, and V is a matrix containing the right singular vectors of X. Applying SVD to the reshaped digit images allows me to identify the most important features of the digit images, which heps with classification and clustering.

### Task 2)
What does the singular value spectrum look like and how many modes are necessary for good
image reconstruction? (i.e. what is the rank r of the digit space?)
```
plt.plot(S)
plt.xlabel('Singular Value Index')
plt.ylabel('Singular Value')
plt.show()
```
To show the singular value spectrum and see how many modes are necessary for good image reconstruction, I plotted the singular values, S that were obtained from the SVD using matplotlib functions above. The singular values of S obtained from the SVD are used to determine the importance of each singular vector in the decomposition of the matrix X, identify the most significant features of the data, and to determine the number of modes that are needed for good image reconstruction. 

### Task 3)

What is the interpretation of the U, Σ, and V matrices?
As stated before, the U matrix contains the left singular vectors, which capture patterns and variability in the rows of the data matrix. The Σ (S) matrix contains the singular values, which indicate the importance or significance of each singular vector. The V matrix contains the right singular vectors, which capture patterns and variability in the columns of the data matrix.

To interpret U and V, they were plotted using the code below: 
```
plt.scatter(U[:, 0], U[:, 1])
plt.xlabel('First Left Singular Vector')
plt.ylabel('Second Left Singular Vector')
plt.title('Left Singular Vectors')
plt.show()

# Plot the right singular vectors
plt.scatter(V[0, :], V[1, :])
plt.xlabel('First Right Singular Vector')
plt.ylabel('Second Right Singular Vector')
plt.title('Right Singular Vectors')
plt.show()
```
The left singular vectors (U) are plotted using the first two columns of the U matrix and the right singular vectors(V) are plotted using the first two rows of the V matrix. The scatter plots show the relationship between the different components of the left and right singular vectors.

### Task 4)
On a 3D plot, project onto three selected V-modes (columns) colored by their digit label. For
example, columns 2,3, and 5.

```
# Select the 2nd, 3rd, and 5th V-modes
V_select = V[:, [2, 3, 5]]  
```
To start I select the 2nd, 3rd, and 5th columns of the right singular vectors matrix V and store them in a new matrix ```V_select```. 

```
X_proj = np.dot(X, V_select)
print(X_proj.shape)
```
I then projected the data onto the selected V-modes using matrix using matrix multiplication and checked that the shape was correct. I had to check the shape of ```X_proj``` to make sure that the projection was performed correctly and the matrix had the right dimensions, (10000, 3). 10000 is the number of datapoints, and 3 is the selected 3 V-modes from ```V_select```. 

```
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_proj[:, 0], X_proj[:, 1], X_proj[:, 2], c=Y.astype(int))
plt.colorbar(scatter)
ax.set_xlabel('V-mode 2')
ax.set_ylabel('V-mode 3')
ax.set_zlabel('V-mode 5')
plt.show()
```
Lastly, I created a 3D scatter plot colored by digit label. The ```fig.add_subplot()``` function is used to create the 3D plot, and the scatter function is used to plot the points. The ```c``` argument of the scatter function is set to the array of digit labels ```Y```, which is cast to an integer data type using the ```astype``` function. This ensures that each digit is assigned a unique color. Finally, the ```ax.set_xlabel()```, ```ax.set_ylabel()```, and ```ax.set_zlabel()``` functions are used to set the labels for the x-axis, y-axis, and z-axis of the plot, respectively. The ```plt.colorbar function``` is used to add a color bar to the plot that shows the correspondence between digit labels and colors. The resulting plot can be used to visualize how the selected V-modes capture the variability in the data and how well the digit images can be separated based on their labels.

### Now, the task is to build a classifier to identify individual digits in the training set

### Task 5)
Pick Two Digits and build a Linear Classifier (LDA) that can reasonably identify/ classify them

To start, I split the data into training and testing sets:
```
X_train, X_test, Y_train, Y_test = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=42)
```
The ```train_test_split function``` is being used here from the scikit-learn library to randomly split the MNIST dataset into the training and testing subsets. The ```mnist.data``` and ```mnist.target``` arguments are the input features and target labels of the dataset, respectively. The ```test_size``` argument specifies the proportion of the dataset that should be used for testing. In this case, it's set to 0.2, which means 20% of the data will be used for testing. The ```random_state``` argument sets the random seed to ensure that the same random split is generated each time the code is run.

Then, I picked two digits (2 and 3) and created a new dataset with only those two digits
```
X_train_2 = X_train[(Y_train == '2') | (Y_train == '3')]
Y_train_2 = Y_train[(Y_train == '2') | (Y_train == '3')]

X_test_2 = X_test[(Y_test == '2') | (Y_test == '3')]
Y_test_2 = Y_test[(Y_test == '2') | (Y_test == '3')]
```

Using the new training dataset, it was fit on a linear discriminant analysis model.
```
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_2, Y_train_2)
```

The ```LinearDiscriminantAnalysis()``` class is a model that learns the discriminative information between the classes in the training data by modeling the class-conditional densities. Once the model is trained on the training data, it can then be used to predict the class labels of new, unseen data.

Predictions are then made on the test and training data
```
Y_pred = lda.predict(X_test_2)
Y_pred2 = lda.predict(X_train_2)
```

These two lines of code are used to generate predictions on the test and training data using the fitted LDA model.
```lda.predict(X_test_2)``` predicts the class labels of the test dataset ```X_test_2``` using the fitted LDA model. The predicted class labels are stored in the variable ```Y_pred```. The same is done for ```Y_pred2```, using the training dataset ```X_train_2```. These predicted labels can then be used to calculate the accuracy of the model on both the training and test data.

The accuracies of the test data and training data are computed and then printed
```
accuracy1 = accuracy_score(Y_test_2, Y_pred)
accuracy2 = accuracy_score(Y_train_2, Y_pred2)
print("Test Accuracy:",accuracy1)
print("Training Accuracy:",accuracy2)
```
The ```accuracy_score``` function takes two arguments, the actual labels ```Y_test_2``` / ```Y_train_2``` and predicted labels ```Y_pred``` / ```Y_pred2```. It then calculates the accuracy of the model by comparing the actual labels with the predicted labels.

### Task 6)
Now, pick Three Digits and build a Linear Classifier (LDA) that can reasonably identify/ classify them

The following code structure from the previous task is then repeated, but now using the three digits 1, 4, and 8:

```
# Create a new dataset with only the two digits
X_train_3 = X_train[(Y_train == '1') | (Y_train == '4')| (Y_train == '8')]
Y_train_3= Y_train[(Y_train == '1') | (Y_train == '4')| (Y_train == '8')]

X_test_3 = X_test[(Y_test == '1') | (Y_test == '4') | (Y_test == '8')]
Y_test_3 = Y_test[(Y_test == '1') | (Y_test == '4') | (Y_test == '8')]

# Fit a linear discriminant analysis model to the data
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_3, Y_train_3)

# Make predictions on the test data
Y_pred = lda.predict(X_test_3)
Y_pred2 = lda.predict(X_train_3)

# Compute the accuracy of the predictions
accuracy = accuracy_score(Y_test_3, Y_pred)
accuracy2 = accuracy_score(Y_train_3, Y_pred2)
print("Test Accuracy:",accuracy)
print("Training Accuracy:",accuracy2)
```

### Task 7 and 8)
Which Two Digits in the Data Set are the most Easy and Most Difficult to Separate? Quantify the accuracy of the seperation with LDA on the test data

To start, I created a list ```accuracies``` to store the accuracies of all the two digit combinations. I then use a double forloop to iterate through every number combination: 
```
accuracies = []

for i in range(10):
    for j in range(10):
        if i != j:
            X_train_2 = X_train[(Y_train == str(i)) | (Y_train == str(j))]
            Y_train_2 = Y_train[(Y_train == str(i)) | (Y_train == str(j))]

            X_test_2 = X_test[(Y_test == str(i)) | (Y_test == str(j))]
            Y_test_2 = Y_test[(Y_test == str(i)) | (Y_test == str(j))]

            # Fit a linear discriminant analysis model to the data
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_train_2, Y_train_2)

            # Make predictions on the test data
            Y_pred = lda.predict(X_test_2)

            # Compute the accuracy of the predictions
            accuracy = accuracy_score(Y_test_2, Y_pred)

            print("The accuracy for digits", i, "and", j, "is", accuracy)

            # Append the i, j values and accuracy to the list
            accuracies.append((i, j, accuracy))
```

In this double forloop, the same steps as the previous tasks are done - a linear discriminant model is fitted on the data, predictions are made on the test data, and the accuracy of the predictions are computed and predicted. The only difference is that this is done for every number combination test data and added to a list, not just two numbers that are selected randomly. 

Find the tuple with the highest accuracy
```
max_accuracy = max(accuracies, key=lambda x: x[2])
```
Find the tuple with the lowest accuracy
```
min_accuracy = min(accuracies, key=lambda x: x[2])
```
To find the tuples with the highest and lowest accuracy, I used the ```max()``` and ```min()``` functino to find the values with the lowest and highest accuracy values.

```
print("The lowest accuracy is for digits", min_accuracy[0], "and", min_accuracy[1], "with an accuracy of", min_accuracy[2])
print("The highest accuracy is for digits", max_accuracy[0], "and", max_accuracy[1], "with an accuracy of", max_accuracy[2])
```
I then printed the i, j values and accuracy with the lowest/ highest accuracy and hardest/ easiest to seperate respectively

### Task 9)
SVM (Support Vector Machines) and Decision Tree Classifiers were the State-of-the-Art until about 2014. How Well do these Seperate Between all Ten Digits?

I started by splitting the data into training and testing sets like before: 
```
X_train, X_test, Y_train, Y_test = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=42)
```

I then created a support vector machine classifier
```
svm = SVC()
```

Now, I fit the classifier to the training data
```
svm.fit(X_train, Y_train)
```

Make predictions on the test data
```
Y_pred = svm.predict(X_test)
```
Computing the accuracy of the predictions and printing
```
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy of the SVM:", accuracy)
```

Now, I create a decision tree classifier below and fit the classifier to the training data
```
tree = DecisionTreeClassifier()
tree.fit(X_train, Y_train)
```

Make predictions on the test data
```
Y_pred = tree.predict(X_test)
```
Compute the accuracy of the predictions and print 
```
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy of the Decision Tree Classifier:", accuracy)
```
The accuracies of how well all ten digits are seperated between the SVM and the Decision Tree Classifier are printed to compare which results in higher accuracy.

To visualize the Decision Tree, it is printed using the code below:
```
fig, ax = plt.subplots(figsize=(10, 10))
plot_tree(tree, ax=ax)
plt.show()
```

### Task 10)
Compare the Performance between LDA, SVM, and Decision Trees on the hardest and easiest pair of Digits to Seperate (from above)

Again, we create the dataset with only the two digits, with the digits that are the hardest to seperate
```
X_train_2 = X_train[(Y_train == '5') | (Y_train == '8')]
Y_train_2 = Y_train[(Y_train == '5') | (Y_train == '8')]
X_test_2 = X_test[(Y_test == '5') | (Y_test == '8')]
Y_test_2 = Y_test[(Y_test == '5') | (Y_test == '8')]
```

Then fit a linear discriminant analysis model to the data
```
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_2, Y_train_2)
```
Make predictions on the test data
```
Y_pred = lda.predict(X_test_2)
```
Compute the accuracy of the predictions and print it
```
accuracy = accuracy_score(Y_test_2, Y_pred)
print("The accuracy of the linear discriminant analysis model (LDA) on the hardest digits to seperate (5 and 8) is", accuracy)
```

Now, fit a support vector machine classifier to the data
```
svm = SVC()
svm.fit(X_train_2, Y_train_2)
```
Make predictions on the test data
```
Y_pred = svm.predict(X_test_2)
```
Compute and print the accuracy of the predictions
```
accuracy = accuracy_score(Y_test_2, Y_pred)
print("The accuracy of the SVM model on the hardest digits to seperate (5 and 8) is", accuracy)
```

Now, fit a decision tree classifier to the data
```
tree = DecisionTreeClassifier()
tree.fit(X_train_2, Y_train_2)
```

Make predictions on the test data
```
Y_pred = tree.predict(X_test_2)
```

Compute the accuracy of the predictions and print
```
accuracy = accuracy_score(Y_test_2, Y_pred)
print("The accuracy of the Decision Tree Classifier model on the hardest digits to seperate (5 and 8) is", accuracy)
```
This process was then repeated on the easiest pair of Digits to Seperate, which are 6 and 7. 

## Sec. IV Computational Results

### Task 1)
U is a matrix containing the left singular vectors of X, These vectors span the column space of X and are orthogonal to each other.
S is a diagonal matrix containing the singular values of X. These values represent the importance of each singular vector in the decomposition of X.
V is a matrix containing the right singular vectors of X. These vectors span the row space of X and are also orthogonal to each other.
By applying the SVD to the transposed matrix of reshaped digit images, you are essentially representing the digit images as linear combinations of the left singular vectors, weighted by the singular values. This allows you to identify the most important features of the digit images, which can be useful for tasks such as classification, clustering, and dimensionality reduction.
### Task 2)
### Task 3)
In other words, the U, Σ, and V matrices together provide a decomposition of the original data matrix into lower-dimensional subspaces that capture different aspects of the underlying structure and patterns in the data. This decomposition can be used for a variety of purposes, such as dimensionality reduction, data compression, and feature extraction.

Based on your description of the plots, it appears that the left singular vectors (U) of the MNIST dataset are spread out as they go left, while the right singular vectors (V) form a large cluster. This suggests that the left singular vectors capture variability in the rows of the data matrix (i.e., the individual images of handwritten digits), while the right singular vectors capture variability in the columns of the data matrix (i.e., the individual pixels within each image).

More specifically, the spread-out pattern of the left singular vectors suggests that they may capture patterns related to the shape, orientation, or style of the handwritten digits, while the clustering of the right singular vectors suggests that they may capture patterns related to the intensity, contrast, or positioning of the pixels within the images.

Overall, the specific interpretation of the U, Σ, and V matrices in the context of the MNIST dataset would require further analysis and investigation, but the observed patterns in the plots suggest that the SVD is capturing important features and variability in the dataset.

### Task 4)


Each point in the plot represents one digit image, and the position of the point in the 3D space is determined by the projection of that image onto the selected V-modes. The color of the point corresponds to the label of the digit, with each digit being assigned a unique color.
### Task 5)
### Task 6)
### Task 7)
### Task 8)
### Task 9)
## Sec. V Summary and Conclusions



