# EE399 Homework 1: Model Fitting with Least Square Error

### Author : Hana McVicker

EE399: Overview of Machine Learning Method with a Focus on Formulating the Underlying Optimization Problems Required to Solve Engineering and Science Related Problems

## Abstract
This homework involves fitting a cosine function with linear and constant terms to a dataset using least-squares error. The code finds the minimum error and determines the parameters, and the results are then used to create a 2D loss landscape by fixing two of the parameters and sweeping through values of the other two parameters. The minimum errors for line, parabola, and 19th-degree polynomial fits on the training data are also obtained, and the models are tested on the remaining 10 data points. The process is repeated using only the first and last 10 data points as training data, and the results are compared to the first method.

## Sec. I. Introduction and Overview

In this assignment, a dataset with 31 points is given, and the focus is on exploring the impact of least-squares error on model fitting. A cosine function is used with linear and constant terms to fit the data while minimizing the error between the model predictions and the actual data.

After finding the minimum error and the optimal values of the parameters, two of the parameters will be fixed, and the other two parameters will be varied to generate a 2D loss (error) landscape. This will help to visualize how the error changes with different parameter values and identify any minima in the loss landscape.

Furthermore, the first 20 data points will be used as training data to fit a line, parabola, and 19th-degree polynomial models, and the least-square error will be calculated for each model over the training points. The models' performance will be evaluated by computing the least square error of these models on the test data (the remaining 10 data points). The same process will be repeated using only the first and last 10 data points as training data, and the results will be compared to the first method.

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
 
 ### Question I
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
   ```
    plt.plot(np.arange(0, 31), Y, 'ko', label = "Given Data")
    plt.plot(tt, yfit, 'r-', label = "Model Fit")
    plt.title('Least Square Error Fit')
    plt.xlabel('X ')
    plt.ylabel('Y')
    plt.legend(loc='upper center')
    plt.show()
   ```
   This code generates a plot of the original data points and the fitted curve.
   
   In the first line, the code plots the original data points 'Y' using black circles, where 'ko' stands for black circles and the label is generated for the graph.
   
   The np.arange(0, 31) function generates an array of x-values ranging from 0 to 30

   In the second line,the code plots the fitted curve using a red solid line ('r-') wit label "Model Fit"
   
   The yfit variable contains the fitted y-values for each x-value in tt

   In the last lines of code, a title, x and y labels, and legend is called to display the plot on the screen using plt.show()
   
 ### Question II
 With the results of (i), we need to fix two of the parameters and sweep through values of the
 other two parameters to generate a 2D loss (error) landscape, and do every combination of two sweeped parameters and two fixed

To start this question, the initial step was to define the parameter values. This was achieved through the implementation of the ```np.linspace()``` function, which allows one to specify the starting and ending values for each parameter, as well as the number of values to generate. The values for each parameter were selected to ensure that the graph would clearly depict the minima. Specifically, we chose to sweep 100 points for each of the parameters A, B, C, and D, as shown below 

```
A_vals = np.linspace(0, 30, 100)
B_vals = np.linspace(-20, 8, 100)
C_vals = np.linspace(-10, 15, 100)
D_vals = np.linspace(20, 75, 100)
```

After the parameter values are defined, a loss matrix is created, initially set to all zeros, and is named ```loss```
Then, with a nested forloop, we are able to loop though the parameter values and calculate the loss: 
```
for i in range(len(A_vals)):
    for j in range(len(B_vals)):
        a = A_vals[i]
        b = B_vals[j]
        c = C
        d = D
        # calculate the loss for each combination of parameter values
        loss[i,j] = tempfit([a, b, c, d], X, Y)
```
The nested forloop iterate through every combination of indices 'i' and 'j', which correspond to the rows and columns in the ```loss``` array
values 'a' and 'b' are set to the parameters defined above, which allows the array to iterate through every value specified in those variables, effectively  "sweeping" A and B while keeping C and D fixed. The loss is then calculated for each combination of parameter values using the tempfit (loss) function created in the previous queston. The ```loss``` matrix now contains the loss at every combination of indices, effectively creating a 2D loss landscape.The landscape created is then plotted using the  multiple Matplotlib functions. 
```
plt.pcolor(A_vals, B_vals, loss)
plt.colorbar()
plt.xlabel('A')
plt.ylabel('B')
plt.title('Fixed C and D, Sweeping A and B')
plt.show()
```
The current code generates a loss landscape for a fixed C and D parameter, while sweeping the A and B parameters. To obtain a complete 2D loss landscape for all combinations of two fixed parameters and two sweeped parameters, the code needs to be repeated five more times. This can be achieved by fixing two parameters and sweeping the other two, for each of the remaining parameter combinations. To implement this, the existing code can be modified by changing the parameter values accordingly. The previously defined np.linspace() function would be used to specify the values to be swept, while the fixed values would be kept the same for each combination.By repeating this process for each parameter combination, a complete 2D loss landscape for all possible parameter values can be obtained.

 ### Question III
 Using the first 20 data points as training data, this task requires us to fit a line, parabola, and 19th degree polynomial to the data and compute the least-square error for each of these over the training points. Then, compute the least square error of these models on the test data, which are the remaining 10 data points. 
 
 To implement this, I started by splitting the array so that my training data only contained the first 20 data points of the original data:
 ```
training_data_x = X[:20]
training_data_y = Y[:20]
```
Now I can use the training data to fit a line, parabola, and 19th degree polynomial to the data. 
 ```
def linefit(c, X, Y):
    # 
    model = c[0] * X + c[1]
    e2 = np.sqrt(np.sum((model - Y)**2) / 20)
    return e2

def parabolafit(c, X, Y):
    # 
    model = c[0] * X**2 + c[1]* X + c[2]
    e2 = np.sqrt(np.sum((model - Y)**2) / 20)
    return e2

def polyfit(c, X, Y):
    model = np.polyval(c, X)
    e2 = np.sqrt(np.sum((model - Y)**2) / 20)
    return e2
```

For each fit, three different functions were created: The first function is for a line, which uses a linear model ```c[0] * X + c[1]```. The other two functions also have a defined model, except the ```parabolafit``` model uses a quadratic equation and the ```polyfit``` function uses a polynomial equation using the ```np.polyval()``` function which calls 'c' and 'X', where 'c' is the coefficients  of the polynomial in decreasing order and 'X' is the values at which to evaluate the polynomial (in this case, the 20 training data points). The error ```e2``` is then calculated using the 'Y' datapoints and the values that are obtained from each model equation. 


To get the error of each fit, the coefficients of each fit are needed. 
The code below shows how the coefficents were extracted for the line fit, and then used to find the error:
```
#Line Coefficients 
coeff_line = np.polyfit(training_data_x, training_data_y, 1)
# calculate line error
line_error = linefit(coeff_line, training_data_x, training_data_y)
print("Minimum Line Error = " + str(line_error))
```
In the code, the coefficents are extracted using the ```np.polyfit()``` function,  where the '1' represents the degree of the polynomial for a line. 
The line error is then calculated using the previously made functions that calculate the error and printed.

This process was repeated for the parabola fit and polynomial fit, where the only difference was the coefficients that were extracted had a different degree of freedom ( parabola had a DOF = 2, polynomial had a DOF = 19). 

**Now, the least square error of these models are computed on test data with the remaining data points:**

 To implement this, I started by splitting the array so that the testing data only contained the remaining data points of the original data:
 ```
testing_data_x = X[20:]
testing_data_y = Y[20:]
``` 
To compute the least square error of these models on test data, the coefficients that were previously computed with the training data are used
with the test data in the fit functions to get the fit errors of the test data. The errors are then printed for each fit.
```
linetest_error = linefit(coeff_line, testing_data_x, testing_data_y)
print("Minimum Line Error for Test = " + str(linetest_error))

parabolatest_error = parabolafit(coeff_parabola, testing_data_x, testing_data_y)
print("Minimum Parabola Error for Test = " + str(parabolatest_error))

polytest_error = polyfit(coeff_poly, testing_data_x, testing_data_y)
print("Minimum Polynomial Error for Test = " + str(polytest_error))
``` 
 ### Question IV
 For the last question, the previous question (III) was repeated, but instead of using the first 20 data points as training data and the last 11 points as testing data, the first 10 and last 10 data points are used as training data and the testing data was the 11 middle data points. 
 
 To get the right training data, the values of the first ten and last ten numbers needed to be stored in a single array. To do this, 
 the first and last ten numbers were stored in arrays for the X and Y data, and then concatenated into a single array: 
 
 ```
first_tenX = X[:10]
last_tenX = X[-10:]

first_tenY = Y[:10]
last_tenY = Y[-10:]

X_data = np.concatenate((first_tenX, last_tenX))
Y_data = np.concatenate((first_tenY, last_tenY))

training_data_x = X_data
training_data_y = Y_data
``` 

To get the testing data, the ten middle values of the original data were extracted:
```
testing_data_x = X[10:20]
testing_data_y = Y[10:20]
``` 

With this training and testing data, the same exact steps from question three were done again, this time using the new training and testing data set. 

## Sec. IV. Computational Results
The first task was to fit the model below to the data with least-squares error:

<img width="245" alt="image" src="https://user-images.githubusercontent.com/72291173/230575877-db9e596a-7086-44dd-8525-5a16402d8900.png">

Using the code from the previous section, this graph was configured to show the model fit on the data:

<img width="639" alt="image" src="https://user-images.githubusercontent.com/72291173/230860554-9bd3bc87-837b-42d0-8897-8fc776ef602b.png">

### Question I:
The results for this question was finding minimum error and determine the parameters A, B, C, D
Using the implemention described in the previous section, the output result is shown below:
```
Minimum Error = 1.5927258505678883
Parameters for A, B, C, D respectively = [ 2.1716925   0.90932536  0.73248796 31.45279766]
```
### Question II:
With the results of (i), generate a 2D loss (error) landscape with two fixed parameters and two sweeping parameters. The results for each combination are shown below:

<img width="567" alt="image" src="https://user-images.githubusercontent.com/72291173/230862897-54e0b5f4-ec26-47c1-a326-e57dd728d7ee.png">

<img width="575" alt="image" src="https://user-images.githubusercontent.com/72291173/230863007-c71223da-45e3-4746-9e9e-2a108e0bf50c.png">

<img width="570" alt="image" src="https://user-images.githubusercontent.com/72291173/230863146-aa7d2df0-c52b-4cdb-870f-e8c5ba983bf1.png">

<img width="541" alt="image" src="https://user-images.githubusercontent.com/72291173/230863421-9a0e6c56-38e7-4ff9-aded-c321906616c4.png">

<img width="551" alt="image" src="https://user-images.githubusercontent.com/72291173/230863667-d6429e89-ee57-44dd-a61d-613b24846ac6.png">

<img width="574" alt="image" src="https://user-images.githubusercontent.com/72291173/230863773-c09e7e4d-5fc8-4849-b76c-bcacbb9042eb.png">

How many minima can you find as you sweep through
There are many minimas found as you sweep though the different parameters

### Question III:
 The least-squares error computed for each of the fits for the testing and training data are shown below:
 
 <img width="417" alt="image" src="https://user-images.githubusercontent.com/72291173/230867890-a0132018-33f6-47a9-a42b-ef567d234042.png">
 
 <img width="483" alt="image" src="https://user-images.githubusercontent.com/72291173/230871094-08dc1b0c-ddd3-430e-8a12-90d382ddc586.png">
 
 ### Question IV:
 The least-squares error computed for each of the fits for the testing and training data are shown below:
 
<img width="412" alt="image" src="https://user-images.githubusercontent.com/72291173/230871462-f5368c55-8f0e-4ade-bdcc-e1b20a2f52d5.png">

<img width="477" alt="image" src="https://user-images.githubusercontent.com/72291173/230871511-8d42eeaf-9443-4ace-9a43-bb8af39100e8.png">

## Sec. V. Summary and Conclusions
