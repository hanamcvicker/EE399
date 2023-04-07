# EE399 Homework 1: Model Fitting with Least Square Error

### Author : Hana McVicker

EE399: Overview of Machine Learning Method with a Focus on Formulating the Underlying Optimization Problems Required to Solve Engineering and Science Related Problems

## Abstract

This homework involves fitting a cosine function with linear and constant terms to a dataset using least-squares error. The code finds the minimum error and determines the parameters. The results are then used to create a 2D loss landscape by fixing two of the parameters and sweeping through values of the other two parameters. The minimum errors for line, parabola, and 19th-degree polynomial fits on the training data are also computed, and the models are tested on the remaining 10 data points. The process is repeated using only the first and last 10 data points as training data, and the results are compared to the first method.

## Sec. I. Introduction and Overview

In this assignment, a dataset with 31 points is given, and the focus is on exploring the impact of least-squares error on model fitting. A cosine function is used with linear and constant terms to fit the data while minimizing the error between the model predictions and the actual data.

After finding the minimum error and the optimal values of the parameters, two of the parameters will be fixed, and the other two parameters will be varied to generate a 2D loss (error) landscape. This will help to visualize how the error changes with different parameter values and identify any minima in the loss landscape.

Furthermore, the first 20 data points will be used as training data to fit a line, parabola, and 19th-degree polynomial models, and the least-square error will be calculated for each model over the training points. The models' performance will be evaluated by computing the least square error of these models on the test data (the remaining 10 data points). The same process will be repeated using only the first and last 10 data points as training data, and the results will be compared to the first method.

The objective of this exercise is to understand the impact of least-squares error on model fitting and explore different methods for fitting models to data. This analysis will provide a deeper understanding of selecting and evaluating models for different types of datasets.

## Sec. II. Theoretical Background

Least-squares regression is a common method for fitting mathematical models to data. Given a set of data points, the goal is to find the best-fitting model that describes the relationship between the independent and dependent variables. Least-squares error is a widely used technique for finding the best-fitting line or curve that describes the relationship between the independent and dependent variables in a set of data. This method involves minimizing the sum of the squared differences between the predicted values of the model and the actual data points:

<img width="309" alt="image" src="https://user-images.githubusercontent.com/72291173/230564742-09a7aada-0737-4f44-bbd4-547ce6c085da.png">

The method is based on the principle that the best estimate for the model parameters should be the one that minimizes the sum of the squared differences between the predicted values and the actual data points. In other words, the model parameters that minimize the least-squares error are the ones that best fit the data. The least-squares method involves minimizing the sum of the squared differences between the predicted values of the model and the actual data points.

## Sec. III. Algorithm Implementation and Development

**Given Data :**
```
X=np.arange(0,31)

Y=np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
```
The first task was to fit the following model below to the data with least-squares error: 

<img width="245" alt="image" src="https://user-images.githubusercontent.com/72291173/230575877-db9e596a-7086-44dd-8525-5a16402d8900.png">

We are able to fit the model to the data with least-squares error function 'tempfit' below.

```
def tempfit(c, X, Y):
    model = c[0] * np.cos(c[1] * X) + c[2] * X + c[3]
    e2 = np.sqrt(np.sum((model - Y)**2) / 31)
    return e2
 ```
 
 The tempfit function uses the least-squares error method to fit a model to a given set of data. The model is the cosine function given above, and the function computes the least-squares error between the model predictions and the actual data points using the formula np.sum((model - Y)**2) / 31.
 The function returns the value of the minimum error, e2. By adjusting the values of the coefficients c, the function can find the best-fitting model that describes the relationship between the independent and dependent variables in the data.
 
 ## Question I
 The first question of this assignment asks to find the minimum error and determine the parameters A, B, C, D in our model function. 
 
 The following Steps are below:
 - **Perform Optimization**
   ```
   res = opt.minimize(tempfit, c0, args=(np.arange(0, 31), Y), method='Nelder-Mead')
   ```
    This code uses the minimize function from the scipy.optimize module to find the best coefficient values of the c parameter vector that minimize the         least-squares error for the given data. 
    
    The tempfit function is the function to minimize and c0 as the initial guess for the c parameter vector.
    
    The args parameter is the X values ( using np.arange(0, 31)) and the Y values, which are the actual data points.
    
    The method parameter specifies the optimization algorithm to use, which is the Nelder-Mead method.
    
 - **Get the Optimized Parameters for A, B, C, and D**
   ```
   c = res.x
   minimum_error = tempfit(c, X, Y)
   ```
   After running the minimize function, the coefficient values are stored in 'c'. 
   
   To obtain the minimum error, the tempfit function is called again using these optimized coefficients, along with the X and Y data.
   
   The tempfit function calculates the least-squares error based on our model function and Y data
   
 - **Generate the Data for Plotting**
   ```
   tt = np.arange(0, 31.01, 0.01)
   yfit = c[0] * np.cos(c[1] * tt) + c[2] * tt + c[3]
   ```
   The 'tt' variable generates x-values between 0 and 31 which are incremented by 0.01. 
   
   Then, the fitted model is evaluated at each 'tt' value using the coefficients stored in c.

   The equation of the fitted model is 'yfit', which contains the optimized coefficients substituted for c0 (our initial guess coefficients).

 - **Plot the Raw Data and the Fitted Curve**
 
# Plot the Raw Data and the Fitted Curve
  ```
   plt.plot(np.arange(0, 31), Y, 'ko')
   plt.plot(tt, yfit, 'r-')
   plt.show()
   ```
   This code generates a plot of the original data points and the fitted curve.
   
   In the first line, the code plots the original data points 'Y' using black circles, where 'ko' stands for black circles.
   
   The np.arange(0, 31) function generates an array of x-values ranging from 0 to 30

   In the second line,the code plots the fitted curve using a red solid line ('r-'). 
   
   The yfit variable contains the fitted y-values for each x-value in tt

   In the last line of code, 'plt.show()' is called to display the plot on the screen.
   
 ## Question II
 With the results of (i), we need to fix two of the parameters and sweep through values of the
 other two parameters to generate a 2D loss (error) landscape, and. do every combination of two sweeped parameters and two fixed
 
 # set the fixed values for parameters
A = c[0]
B = c[1]
C = c[2]
D = c[3]

# define the parameter values to sweep
A_vals = np.linspace(0.1, 3, 50)
B_vals = np.linspace(0.1, 1, 50)
C_vals = np.linspace(0.1, .8, 50)
D_vals = np.linspace(0.1, 32, 50)


## Fixed C and D ##
# initialize the loss matrix
loss = np.zeros((len(A_vals), len(B_vals)))

# loop through the parameter values and calculate the loss
for i in range(len(A_vals)):
    for j in range(len(B_vals)):
        a = A_vals[i]
        b = B_vals[j]
        c = C
        d = D
        # calculate the loss for each combination of parameter values
        loss[i,j] = tempfit([a, b, c, d], X, Y)

# plot the 2D loss landscape using pcolor
plt.pcolor(A_vals, B_vals, loss)
plt.colorbar()
plt.xlabel('A')
plt.ylabel('B')
plt.title('Fixed C and D, Sweeping A and B')
plt.show()

