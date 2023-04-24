# EE399 Homework 2

### Author : Hana McVicker

EE399: Overview of Machine Learning Method with a Focus on Formulating the Underlying Optimization Problems Required to Solve Engineering and Science Related Problems

## Abstract
In this assignment, the task was to analyze a dataset of 2414 images of 39 different faces with 65 lighting scenes for each face. A correlation matrix was computed between the images, and the most highly correlated and uncorrelated images are identified. Eigenvectors and principal components are also found using matrix operations. The first eigenvector is compared to the first SVD mode, and the percentage of variance captured by each of the first six SVD modes is computed and plotted.

## Sec. I Introduction and Overview

This assignment involves analyzing the dataset, with each image downsampled to 32x32 pixels. The main objectives of the assignment are to compute the correlation matrix between the images, find the most highly and uncorrelated images, compute eigenvectors and principal components, and compare the first eigenvector to the first SVD mode. The percentage of variance captured by each of the first six SVD modes is also calculated and plotted.  These analyses aim to uncover patterns and structures within the dataset and provide insights into the significant variations in the images.

## Sec. II Theoretical Background

Correlation matrices show how closely different variables in a dataset are related to each other. They are used to study the relationships between variables and can help identify groups of related variables. Singular value decomposition (SVD) is a way to break down a matrix into its principal components, which are the most important patterns of variation in the data. This can be useful for finding patterns in large datasets or reducing the amount of data needed to represent a system.

## Sec. III Algorithm Implementation and Development
For this assignment, we are given the dataset ```yaleface.mat``` which contains the dataset of the 2414 images with 39 different faces and 65 lighting scenes for each face. 
```
import numpy as np
from scipy.io import loadmat
results=loadmat(’yalefaces.mat’)
X=results[’X’]
```
The individual images are columns of the matrix X, where each image has been downsampled to 32×32
pixels and converted into gray scale with values between 0 and 1, with X as a matrix size 1024 x 2414

### Task (a)
For this task, I needed to compute a 100 x 100 correlation matrix C where I would compute the dot product (correlation) between the first 100 images
in the given matrix X. The correlation matrix is then plotted. 

To start, I needed to select the first 100 images from the matrix X:
```
X_100 = X[:, :100]
```
Then, to compute the correlation matrix C, I used the ```np.corrcoef()``` function below:
```
C = np.corrcoef(X_100, rowvar = False)
```

The ```np.corrcoef``` function takes ```X_100``` and ```rowvar = False```. The ```rowvar``` argument indicates whether each row of ```X_100``` represents a variable (each column is a separate observation) or each column represents a photo/variable (each row is a separate observation). In this case, ```rowvar = False``` specifies that each column of ```X_100``` represents a variable/photo. This results in a symmetric matrix C, which is the correlation coefficient matrix of the given array ```X_100```.  The correlation coefficient is a measure of the linear relationship between two variables, and it ranges from -1 (perfect negative correlation) to 1 (perfect positive correlation), with 0 indicating no correlation.
```
print(C.shape)
```
In this line of code, the shape of C is printed to ensure that the size of the C matrix is 100x100

The Correlation Matrix of the First 100 images are then plotted and labeled using ```matplotlib.pyplot``` functions:
```
# Plot the correlation matrix
plt.pcolor(C)
plt.title('Correlation Matrix of First 100 Images')
plt.xlabel('Image Index')
plt.ylabel('Image Index')
plt.colorbar()
```

### Task (b)
For this task, I take the correlation from part (a) and find which two images are most highly correlated and which two images are most uncorrelated. 
These images are then plotted

To ensure the accuracy of the output for the most highly correlated image, I replaced all occurrences of the value 1 in the correlation matrix C. This was necessary because leaving 1 values in the matrix would result in images that are identical being chosen as the most highly correlated, which is not accurate. The ```np.fill_diagonal()``` function was used to replace the 1 values with -10, since the correlation matrix only contains values between -1 and 1, replacing the 1 values with -10 does not affect the true correlation values.
```
np.fill_diagonal(C, -10)
```
To find the indices of the most uncorrelated and most correlated images, the function below are used: 
```
max_correlated = np.unravel_index(np.argmax(C), C.shape)
min_correlated = np.unravel_index(np.argmin(np.abs(C)), C.shape)
```
The ```np.unravel_index()``` function converts a flattened index into its corresponding coordinates in an array of specified shape. ```np.unravel_index(np.argmax(C) / np.argmin(np.abs(C)), C.shape)``` returns a tuple of two indices corresponding to the location of the maximum/ minimum value in the correlation matrix C, which represents the pair of images that are most highly/least correlated. To get the index wanted for highest and least correlation, the ```np.argmax()``` and ```np.argmin()``` functions are used on the matrix C. Because all of the 1s in the matrix are no longer there, the index pulls the next highest indices that are the most correlated. In the ```np.argmin()``` function, ```np.abs(C)``` is used instead because I wanted to get the value closest to 0, not -1. If the absolute value was not there, the image indices would be the minimum (closest to -1) which is not the least correlated, but are actually a perfect negative correaltion. Gettting the minimum that is closest to 0 indicates the least correlation, which is what I was looking for. 
```
print(max_correlated)
print(min_correlated)
```
The most uncorrelated and most correlated image indices are printed so that they can be labeled on the images

The images are then plotted and labeled using matplotlib.pyplot functions: 
```
# Plot the most highly correlated images
plt.figure()
plt.subplot(1,2,1)
plt.imshow(X_100[:, max_correlated[0]].reshape(32,32), cmap='gray')
plt.title('Most Correlated Image (5)')
plt.xlabel('Image Index')
plt.ylabel('Image Index')
plt.subplot(1,2,2)
plt.imshow(X_100[:, max_correlated[1]].reshape(32, 32), cmap='gray')
plt.title('Most Correlated Image (62)')
plt.xlabel('Image Index')
plt.ylabel('Image Index')

# Plot the most uncorrelated images
plt.figure()
plt.subplot(1,2,1)
plt.imshow(X_100[:, min_correlated[0]].reshape(32, 32), cmap='gray')
plt.title('Most Uncorrelated Image (36)')
plt.xlabel('Image Index')
plt.ylabel('Image Index')
plt.subplot(1,2,2)
plt.imshow(X_100[:, min_correlated[1]].reshape(32,32), cmap='gray')
plt.title('Most Uncorrelated Image (5)')
plt.xlabel('Image Index')
plt.ylabel('Image Index')

```

### Task (c)
For this task, task (a) is repeated, but now the computed correlation matrix is a 10 x 10 correlation matrix between images and plotted.
The image indices I am using are : [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]

To start, I define what indices to extract:
```
indices = [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]
```

I then use slicing to extract a subset of columns from array X:
```
X_10 = X[:, indices]
```

The same steps from task (a) are then repeated to complete the rest of this task, as shown below:
```
C2 = np.corrcoef(X_10, rowvar = False)

# Print the shape of C_10
print(C2.shape)

# Plot the correlation matrix
plt.pcolor(C2)
plt.title('Correlation Matrix of  Images')
plt.xlabel('Image Index')
plt.ylabel('Image Index')
plt.colorbar()
```

### Task (d)
For this task, I created a matrix <img width="75" alt="image" src="https://user-images.githubusercontent.com/72291173/232960384-885e2fb8-956b-4b87-828e-bbcb1e3bf61e.png"> and found the first six eigenvectors with the largest magnitude eigenvalue.

To start, I first used the ```np.matmul()``` function to multiply matrix X and the transpose of matrix X:
```
Y = np.matmul(X,X.T)
```
I then extracted the eigenvalues and eigenvectors using the ```np.linalg``` function ```eig()```:
```
eigenValues, eigenVectors = eig(Y) 
```
The top six eigenvectors and eigenvalues were then extracted by selecting the first six columns (vectors) from the eigenVectors array, and the first six elements (values) from the eigenValues array
```
topsix_Values = eigenValues[:6]
topsix_Vectors = eigenVectors[:, :6]
```

I then printed the first 6 largest magnitude eigenvalues and corresponding Vectors with space in between:
```
print('Six largest Magnitude EigenValue: ' + str(topsix_Values))
print('')
print('First Six Eigenvectors with the Largest Magnitude EigenValue:'+ str(topsix_Vectors))
```
### Task (e)
For this task, I had to SVD the matrix X and find the first six principal component directions: 
```
u, s, v = svd(X)
u_vectors = u[:, :6]
print('First Six Principal Component Directions')
print(u_vectors)
```
Singular value decomposition (SVD) is done on the input matrix X, which decomposes X into three matrices: u, s, and v. The u matrix contains the left singular vectors of X, which represent the directions of maximum variance in the data. The s matrix contains the singular values, which represent the amount of variance explained by each singular vector. The v matrix contains the right singular vectors, which represent the contribution of each original variable to the singular vectors.

After performing SVD, the code selects the first six columns of the u matrix, which correspond to the six largest singular values. This is equivalent to selecting the six left singular vectors that explain the most variance in the data. The vectors are then printed to see the results.

### Task (f)
This task compared the first eigenvector v1 from (d) with the first SVD mode u1 from (e) and computed the 
norm of difference of their absolute values.

The code below compares the first eigevector with the first SVD mode and computes the norm of difference in their absolute values:
```
norm = LA.norm(np.abs(topsix_Vectors[:, 0]) - np.abs(u_vectors[:, 0]))
print('Norm of the Difference of the Absolute Values of the First Eigenvector and SVD Mode u1')
print(norm)
```
The absolute values of the first column of ```topsix_Vectors``` and ```u_vectors``` are first computed using the ```np.abs()``` function. Then, the absolute difference between these two columns is calculated using subtraction. Finally, the norm of this difference vector is calculated using the ```LA.norm()``` function, which takes the vector as an input and returns its magnitude.

### Task (g)
This task computes the percentage of variance captured by each of the first 6 SVD modes and is plotted:
```
variance = np.square(s[:6]) / np.sum(np.square(s))
print('Percentage of Variance Captured by each of the first SVD modes:')
print(variance)
```
In this code, it takes the top 6 singular values (s) and squares them using ```np.square()```. Then it divides the squared singular values by the sum of the squares of all the singular values using ```np.sum(np.square(s))```. This is equivalent to calculating the proportion of total variance explained by the top 6 singular values. The resulting value is assigned to the variable variance and printed to show the results.

The results are plotted and labeled using matplotlib.pyplot functions: 
```
plt.figure()
plt.subplot(2,3,1)
plt.imshow(u_vectors[:, 0].reshape(32,32), cmap='gray')
plt.title('SVD Mode 1')
plt.xlabel('Image Index')
plt.ylabel('Image Index')
plt.subplot(2,3,2)
plt.imshow(u_vectors[:, 1].reshape(32,32), cmap='gray')
plt.title('SVD Mode 2')
plt.xlabel('Image Index')
plt.ylabel('Image Index')
plt.subplot(2,3,3)
plt.subplots_adjust(hspace=0.7)
plt.subplots_adjust(wspace=0.7)
plt.imshow(u_vectors[:, 2].reshape(32,32), cmap='gray')
plt.title('SVD Mode 3')
plt.xlabel('Image Index')
plt.ylabel('Image Index')
plt.subplot(2,3,4)
plt.imshow(u_vectors[:, 3].reshape(32,32), cmap='gray')
plt.title('SVD Mode 4')
plt.xlabel('Image Index')
plt.ylabel('Image Index')
plt.subplot(2,3,5)
plt.imshow(u_vectors[:, 4].reshape(32,32), cmap='gray')
plt.title('SVD Mode 5')
plt.xlabel('Image Index')
plt.ylabel('Image Index')
plt.subplot(2,3,6)
plt.imshow(u_vectors[:, 5].reshape(32,32), cmap='gray')
plt.title('SVD Mode 6')
plt.xlabel('Image Index')
plt.ylabel('Image Index')

```

## Sec. IV Computational Results
### Task (a)
![image](https://user-images.githubusercontent.com/72291173/232963545-bdacce35-b2b2-4334-904e-1d70835be6a0.png)

For this task, I needed to compute a 100 x 100 correlation matrix C where I would compute the dot product (correlation) between the first 100 images
in the given matrix X. As seen in the results, there is a clear diagonal line of 1s in my matrix. This is because the diagonal represents the correlation of each variable with itself, which is always perfectly correlated with itself. So, the values on the diagonal are always 1. This is because the correlation coefficient is a measure of the linear relationship between two variables, and when a variable is compared to itself, there is a perfect linear relationship, resulting in a correlation coefficient of 1.

### Task (b)
For this task, I take the correlation from part (a) and find which two images are most highly correlated and which two images are most uncorrelated.


<img width="635" alt="image" src="https://user-images.githubusercontent.com/72291173/232964205-559f1a1c-97e2-4cd9-a704-c4331bdc459e.png">

<img width="642" alt="image" src="https://user-images.githubusercontent.com/72291173/232964260-03b45711-ff34-45dc-818b-e9cd7b523bc4.png">

As seen by the images, the most correlated iamges are almost identical while the least correlated images that no similarites at all. 
### Task (c)
For this task, task (a) is repeated, but now the computed correlation matrix is a 10 x 10 correlation matrix betwen images and plotted.

<img width="646" alt="image" src="https://user-images.githubusercontent.com/72291173/232964946-90478b87-7718-4ff4-bcbe-11b43751bc38.png">

Again, there is a clear diagonal line of 1s in my matrix, which was also seen above in part (a).

### Task (d)
For this task, I needed to find the first six eigenvectors with the largest magnitude eigenvalue. The results are shown below:

<img width="1011" alt="image" src="https://user-images.githubusercontent.com/72291173/232965576-25ffc50f-fafa-4a54-abea-ceaa2f47ae2d.png">

### Task (e)
For this task, I had to SVD the matrix X and find the first six principal component directions which are shown below:
<img width="651" alt="image" src="https://user-images.githubusercontent.com/72291173/232968436-e3eee9b7-66cc-4943-b5cb-d5a47bcd367f.png">

### Task (f)
This task compared the first eigenvector v1 from (d) with the first SVD mode u1 from (e) and computed the 
norm of difference of their absolute values:

<img width="771" alt="image" src="https://user-images.githubusercontent.com/72291173/232970013-c7ee2a1b-8998-4643-922e-aaf97149faae.png">


### Task (g)
This task computes the percentage of variance captured by each of the first 6 SVD modes and is plotted:

<img width="586" alt="image" src="https://user-images.githubusercontent.com/72291173/232968836-19e7569b-63cd-4025-97b2-38a58cc45848.png">

<img width="746" alt="image" src="https://user-images.githubusercontent.com/72291173/232969501-7d7d38cd-0f02-4c46-b113-df99db0b395d.png">

## Sec. V Summary and Conclusions

This assignment describes the analysis of a dataset consisting of 2414 images of 39 different faces with 65 lighting scenes for each face. The main objective of the analysis is to identify patterns and structures within the dataset and gain insights into the significant variations in the images. This assignment had multiple tasks, including computing the correlation matrix between the images, finding the most highly and uncorrelated images, computing eigenvectors and principal components, and comparing the first eigenvector to the first SVD mode. In conclusion, the analysis of the dataset reveals significant patterns and structures in the images, which can be used to gain insights into the variations in the images. The use of correlation matrices and SVD provides an effective means of identifying these patterns and structures and reducing the amount of data needed to represent the system.
