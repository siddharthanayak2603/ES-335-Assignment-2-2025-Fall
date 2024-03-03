# Assignment 2

Total marks: 11 (This assignment total to 22, we will overall scale by a factor of 0.5)

For all the questions given below, create `assignment_q<question-number>_subjective_answers.md` and write your observations.

## Questions
## Assignment Question

1. Generate the following two functions:

    Dataset 1:
    ```python
    num_samples = 40
    np.random.seed(45) 
        
    # Generate data
    x1 = np.random.uniform(-20, 20, num_samples)
    f_x = 100*x1 + 1
    eps = np.random.randn(num_samples)
    y = f_x + eps
    ```
    
    Dataset 2: 
    ```python
    np.random.seed(45)
    num_samples = 40
        
    # Generate data
    x1 = np.random.uniform(-1, 1, num_samples)
    f_x = 3*x1 + 4
    eps = np.random.randn(num_samples)
    y = f_x + eps
    ```

    - Implement full-batch and stochastic gradient descent. Find the average number of steps it takes to converge to an $\epsilon$-neighborhood of the minimizer for both datasets. Visualize the convergence process for 15 epochs. Choose $\epsilon = 0.001$ for convergence criteria. Which dataset and optimizer takes a larger number of epochs to converge, and why? Show the contour plots for different epochs (or show an animation/GIF) for visualisation of optimisation process. Also, make a plot for Loss v/s epochs. **[2 marks]**
   - Explore the article [here](https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/#:~:text=Momentum%20is%20an%20extension%20to,spots%20of%20the%20search%20space.) on gradient descent with momentum. Implement gradient descent with momentum for the above two datasets. Visualize the convergence process for 15 steps. Compare the average number of steps taken with gradient descent (both variants -- full batch and stochastic) with momentum to that of vanilla gradient descent to converge to an $\epsilon$-neighborhood of the minimizer for both datasets. Choose $\epsilon = 0.001$. Write down your observations. Show the contour plots for different epochs for momentum implementation. Specifically, show all the vectors: gradient, current value of theta, momentum, etc. **[2 marks]**
     
2. Refer to the [instructor's notebook](https://nipunbatra.github.io/ml-teaching/notebooks/dummy-variables-multi-colinearity.html) on multi-colinearity. Use `np.linalg.solve` instead of `np.linalg.inv` for the same problem. Compare and contrast their usage, which one is better and why? **[1 Marks]**

3. Referring to the same [notebook](https://nipunbatra.github.io/ml-teaching/notebooks/dummy-variables-multi-colinearity.html), explain why Sklearn's linear regression implementation is robust against multicollinearity. Dive deep into Sklearn's code and explain in depth the methodology used in sklearn's implementation. **[1 Mark]**
   
4. Begin by exploring the [instructor's notebook](https://github.com/nipunbatra/ml-teaching/blob/master/notebooks/siren.ipynb) that introduces the application of Random Fourier Features (RFF) for image reconstruction. Demonstrate the following applications using the cropped image from the notebook:
    - Superresolution: perform superresolution on the image shown in notebook to enhance its resolution by factor 2. Show a qualitative comparison of original and reconstructed image. **[2 Marks]**
    - The above only helps us with a qualitative comparison. Let us now do a quantitative comparison. First, skim read this article: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution **[2 Marks]**
        - Start with a 400x400 image (ground truth high resolution).
        - Resize it to a 200x200 image (input image)
        - Use RFF + Linear regression to increase the resolution to 400x400 (predicted high resolution image)
        - Compute the following metrics:
            - RMSE on predicted v/s ground truth high resolution image
            - Peak SNR
    - Completing Image with Random Missing Data: Apply RFF to complete the image with 10%, 20%, and so on up to 90% of its data missing randomly. Randomly remove portions of the data, train the model on the remaining data, and predict on the entire image. Display the reconstructed images for each missing data percentage and show the metrics calculated above. What do you conclude?. **[2 Marks]**

5. Use the [instructor's notebook](https://github.com/nipunbatra/ml-teaching/blob/master/notebooks/movie-recommendation-knn-mf.ipynb) on matrix factorisation, and solve the following questions. 

    Image Reconstruction-
    Here, ground truth pixel values are missing for particular regions within the image- you don't have access to them.

    - Use the above image from Q4 and reconstruct the image in the following two cases, where your region is-
        1. a rectangular block of 30X30 is assumed missing from the image. 
        2. a random subset of 900 (30X30) pixels is missing from the image. 
    
        Choose rank `r` yourself. Perform Gradient Descent till convergence, plot the selected regions, original and reconstructed images, compute the metrics mentioned in Q4 and write your observations. 
        Obtain the reconstruction using RFF + Linear regression and compare the two. **[1.5 Marks]**

    - Vary region size (NxN) for ```N = [20, 40, 60, 80]``` and perform Gradient Descent till convergence. Again, consider the two cases for your region as mentioned in Part (a). Demonstrate the variation in reconstruction quality by making appropriate plots and metrics. **[2 Marks]**
            
    - Write a function using this [reference](https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html) and use alternating least squares instead of gradient descent to repeat Part 1, 2 of Q5, using your written function. **[2 Marks]**
    
    Data Compression-
    Here, ground truth pixel values are not missing- you have access to them. You want to explore the use of matrix factorisation in order to store them more efficiently.

    - Consider an image patch of size (NxN) where N=50. We are trying to compress this patch (matrix) into two matrices, by using low-rank matrix factorization. Consider the following three cases-
        1. a patch with mainly a single color.
        2. a patch with 2-3 different colors.
        3. a patch with at least 5 different colors.

     Vary the low-rank value as ```r = [5, 10, 25, 50]```  for each of the cases. Use Gradient Descent and plot the reconstructed patches over the original image (retaining all pixel values outside the patch, and using your learnt compressed matrix in place of the patch) to demonstrate difference in reconstruction quality. Write your observations. **[1.5 Marks]**

<br>

Here is a reference set of patches chosen for each of the 3 cases from left to right. 

<div style="display: flex;">
  <img src="sample_images/1colour.jpg" alt="Image 1" width="250"/>
  <img src="sample_images/2-3_colours.jpg" alt="Image 2" width="270"/>
  <img src="sample_images/multiple_colours.jpg" alt="Image 3" width="265"/>
</div>

<br>

6. UCI-HAR dataset. Compare DT, RF and Linear regression (yes, regression). For linear regression: each class label as an integer value. Say, 1: Sitting, 2:..., and so on. Use features extracted (from flattened out Linear Acceleration) using the TSFEL library. Compare the performance of these models. Is the usage of linear regression for classification justified? Why or why not? **[2 Marks]**

7. Obtain the weights (take absolute values as weights can also be negative) of the linear regression model. Also, obtain the feature importance from the Random Forest model. Plot the weights obtained as a Bar plot. This will help you visualize what features are being prioritized by the models. Note that sum of feature importances for a Random Forest model is 1. you will have to bring the linear regression weights to the same scale. To do so you can divide the weights by the sum of all the weights. Plot the importance of the features in the same plot. Figure out the top 10 important features obtained from both the models and display their names. What do you infer? **[1 Mark]**

