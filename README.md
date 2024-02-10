# Assignment 2

Total marks: 

For all the questions given below, create `assignment_q<question-number>_subjective_answers.md` and write your observations.

## Questions
## Assignment Question

1. Generate the following two functions:

    Dataset 1:
    ```python
    np.random.seed(45) 
        
    # Generate data
    x1 = np.random.uniform(-50, 50, num_samples)
    x2 = np.random.uniform(-50, 50, num_samples)
    y = 100*x1**2 + 2*x2**2 + 9
    ```
    
    Dataset 2: 
    ```python
    np.random.seed(45) 
        
    # Generate data
    x1 = np.random.uniform(-10, 10, num_samples)
    x2 = np.random.uniform(-10, 10, num_samples)
    y = 3*x1**2 + 2*x2**2 + 9
    ```

    - Implement gradient descent. Find the average number of steps it takes to converge to the optimal value for both datasets. Visualize the convergence process. Which dataset takes a larger number of steps to converge, and why?
   - Explore the article [here](https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/#:~:text=Momentum%20is%20an%20extension%20to,spots%20of%20the%20search%20space.) on gradient descent with momentum. Implement gradient descent with momentum for the above two datasets. Visualize the convergence process.
   - Compare the average number of steps taken with gradient descent with momentum to that of vanilla gradient descent.
     
2. Refer to the [instructor's notebook](https://nipunbatra.github.io/ml-teaching/notebooks/dummy-variables-multi-colinearity.html) on multi-colinearity. Use np.linalg.solve instead of np.linalg.inv for the same problem. Compare and contrast their usage, which one is better and why? **[1 Mark]**

3. Referring to the same [notebook](https://nipunbatra.github.io/ml-teaching/notebooks/dummy-variables-multi-colinearity.html), explain why Sklearn's linear regression implementation is robust against multicollinearity. Dive deep into Sklearn's code and explain in depth the methodology used. **[1 Mark]**
   
4. Begin by exploring the [instructor's notebook](https://github.com/nipunbatra/ml-teaching/blob/master/notebooks/siren.ipynb) that introduces the application of Random Fourier Features (RFF) for image reconstruction. Demonstrate the following applications using the cropped image from the notebook:
    - Superresolution: perform superresolution on the image shown in notebook to enhance its resolution by factor 2. **[2 Marks]**
    - Completing Image with Random Missing Data: Apply RFF to complete the image with 10%, 20%, and so on up to 90% of its data missing randomly. Randomly remove portions of the data, train the model on the remaining data, and predict on the entire image. Display the reconstructed images for each missing data percentage. **[1 Mark]**
    - Image Inpainting: Take out a rectangular patch from the image. Utilize RFF to train a linear model using the remaining image data. Reconstruct the entire image using the trained model and highlight the effectiveness of the inpainting process. **[1 Mark]**
5. Use instructor's notebook on matrix factorisation
    - use above image from Q4 and complete the rectangular missing patch [1]
    - write a function using https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html and use alternating least squares instead of gradient descent [1]

6. Create a data set as follows:

```py
np.random.seed(45) 

# Generate data
x = np.array([i*np.pi/180 for i in range(60,300,2)])
y = 3*x + 8 + np.random.normal(0,3,len(x))

# Reshape x to be a column vector
x = x.reshape(-1, 1)

```
Now, using Sklearn's `preprocessing.PolynomialFeatures` generate polynomial features for the data provided above, plot norm of theta (Weights) v/s degree when you fit linear regression using the polynomial of degree `d`. Vary the degree from 1 to 12. What do you observe? What can you conclude? 

7. For the scenario described in the previous question, consider using polynomial features of varying degrees (1, 3, 5, 7, 9). Suppose we vary the size of the dataset, denoted by N, and plot the magnitude of the norm of the weights (theta) against the degree of the polynomial regression model for each value of N. Plot the observed trends and draw conclusions based on the plotted data.

8. Compare the Random Forest algorithm with Decision Tree for recognizing Human Activity using UCI-HAR dataset. We'll test them with tree depths ranging from 2 to 8 and plot their accuracies. Again we'll test using two types of data: one with flattened out linear acceleration data and another with features extracted (from flattened out Linear Acceleration) using the TSFEL library.

9. Use Linear Regression for the above problem. Consider each class label as an integer value. Use featurized data to compare the accuracy obtained from Linear Regression with that of Random Forest and Decision Tree (trees of infinite depth). How did the linear regression perform? Explain why.

10. Obtain the weights(take absolute values as weights can also be negative) of the linear regression model. Also, obtain the feature importance from the Random Forest model. Plot the weights obtained as a Bar plot. This will help you visualize what features are being prioritized by the models. Note that sum of feature importances for a Random Forest model is 1. you will have to bring the linear regression weights to the same scale. To do so you can divide the weights by the sum of all the weights. Plot the importance of the features in the same plot. Figure out the top 10 important features obtained from both the models and display their names.

11. Consider the [Daily Temperatures dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv) from Australia. This is a dataset for a forecasting task. That is, given temperatures up to date (or period) T, design a forecasting (autoregressive) model to predict the temperature on date T+1. You can refer to [link 1](https://www.turing.com/kb/guide-to-autoregressive-models), [link 2](https://otexts.com/fpp2/AR.html) for more information on autoregressive models. Use linear regression as your autoregressive model. Plot the fit of your predictions vs the true values and report the RMSE obtained. A demonstration of the plot is given below. ![imgsrc](./Autoregressive_Demo.png)

