# Assignment 2

Total marks: XX (This assignment total to XX, we will overall scale by a factor of XX)

For all the questions given below, create `assignment_q<question-number>_subjective_answers.md` and write your observations.

## Questions
## Assignment Question

### 1. Understanding Gradient Descent and Momentum [4 Marks]

Generate the following two functions:
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

-  **[2 marks]** Implement full-batch gradient descent and stochastic gradient descent for linear regression using the given datasets. Define the convergence criterion as reaching an $\epsilon$-neighborhood of the minimizer, with $\epsilon = 0.001$. Here, this means that your estimated parameter vector $\theta_t$ is considered to have *converged* once it is within a distance of $\epsilon$ from the true minimizer $\theta^*$. Formally:  $$\|\theta_t - \theta^*\| < \epsilon$$ For each method and dataset, determine the **average number of steps** required to satisfy this convergence criterion. Visualize the convergence process over 15 epochs. Provide visualizations:
    - Contour plots of the optimization process at different epochs (or an animation/GIF).
    - A plot of loss versus epochs for each method and dataset.


- **[2 marks]** Explore the article [here](https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/#:~:text=Momentum%20is%20an%20extension%20to,spots%20of%20the%20search%20space.) on gradient descent with momentum. Implement gradient descent with momentum for the above two datasets. Visualize the convergence process for 15 steps. Compare the average number of steps taken with gradient descent (both variants -- full batch and stochastic) with momentum to that of vanilla gradient descent to converge to an $\epsilon$-neighborhood of the minimizer for both datasets. Choose $\epsilon = 0.001$. Write down your observations. Show the contour plots for different epochs for momentum implementation. Specifically, show all the vectors: gradient, current value of theta, momentum, etc. 


### 2. Effect Of Feature Scaling on Optimisation [2 Marks]

```python
num_samples = 100
np.random.seed(42)

# Generate data with large feature scale
x = np.random.uniform(0, 1000, num_samples)
f_x = 3 * x + 2
eps = np.random.randn(num_samples)
y = f_x + eps
```

- **[1 marks]** Using the above dataset, implement full-batch gradient descent for linear regression on the dataset above without any feature scaling. Define the convergence criterion as reaching an epsilon-neighborhood of the empirical least squares minimizer θ*, with ε = 0.001 ( $\|\theta_t - \theta^*\| < \epsilon$ ). Determine the number of iterations required to satisfy this convergence criterion. Plot mse loss versus iterations plot.

- **[1 marks]** Apply z-score normalization to the feature:  $$ x_{\text{scaled}} = \frac{x - \mu_x}{\sigma_x} $$ Run full-batch gradient descent on the scaled dataset with the same convergence criterion $(\epsilon = 0.001)$. Determine the number of iterations required for convergence. Plot mse loss versus iterations plot.

### 3. Working with Autoregressive Modeling [2 Marks]

- **[2 marks]**  Consider the [Daily Temperatures dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv) from Australia. This is a dataset for a forecasting task. That is, given temperatures up to date (or period) T, design a forecasting (autoregressive) model to predict the temperature on date T+1. You can refer to [link 1](https://www.turing.com/kb/guide-to-autoregressive-models), [link 2](https://otexts.com/fpp2/AR.html) for more information on autoregressive models. Use linear regression as your autoregressive model. Plot the fit of your predictions vs the true values and report the RMSE obtained. A demonstration of the plot is given below. ![imgsrc](./Autoregressive_Demo.png)



### 4. Implementing Matrix Factorization [9 Marks]

Use the [instructor's notebook](https://github.com/nipunbatra/ml-teaching/blob/master/notebooks/movie-recommendation-knn-mf.ipynb) on matrix factorisation, and solve the following questions.

**a) Image Reconstruction-** Here, ground truth pixel values are missing for particular regions within the image- you don't have access to them.

- **[2 Marks]** Use an image and reconstruct the image in the following two cases, where your region is-
    1. a rectangular block of 30X30 is assumed missing from the image. 
    2. a random subset of 900 (30X30) pixels is missing from the image. 

    Choose rank `r` yourself. Perform Gradient Descent till convergence, plot the selected regions, original and reconstructed images, Compute the following metrics:
    * RMSE on predicted v/s ground truth high resolution image
    * Peak SNR

- **[2 Marks]** Vary region size (NxN) for ```N = [20, 40, 60, 80]``` and perform Gradient Descent till convergence. Again, consider the two cases for your region as mentioned in Part (a). Demonstrate the variation in reconstruction quality by making appropriate plots and Computing metrics. 
    
- **[2 Marks]** Write a function using this [reference](https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html) and use alternating least squares instead of gradient descent to repeat Part 1, 2 of Image reconstruction problem using your written function. 

**b) Data Compression-** Here, ground truth pixel values are not missing- you have access to them. You want to explore the use of matrix factorisation in order to store them more efficiently.
- **[3 Marks]** Consider an image patch of size (NxN) where N=50. We are trying to compress this patch (matrix) into two matrices, by using low-rank matrix factorization. Consider the following three cases-
    1. a patch with mainly a single color.
    2. a patch with 2-3 different colors.
    3. a patch with at least 5 different colors.

    Vary the low-rank value as ```r = [5, 10, 25, 50]```  for each of the cases. Use Gradient Descent and plot the reconstructed patches over the original image (retaining all pixel values outside the patch, and using your learnt compressed matrix in place of the patch) to demonstrate difference in reconstruction quality. Write your observations. 

Here is a reference set of patches chosen for each of the 3 cases from left to right. 

<div style="display: flex;">
<img src="sample_images/1colour.jpg" alt="Image 1" width="250"/>
<img src="sample_images/2-3_colours.jpg" alt="Image 2" width="270"/>
<img src="sample_images/multiple_colours.jpg" alt="Image 3" width="265"/>
</div>


### 5. Simulating Epidemic Modeling [2 Marks]

The goal is to simulate and visualize the spread of an epidemic using a linear system based on the SIRD (Susceptible, Infected, Recovered, Deceased) model.

**Background :**
* We will model the proportion of a population distributed across four states:
* Susceptible (S): Individuals who can acquire the disease the next day.
* Infected (I): Individuals who currently have the disease.
* Recovered (R): Individuals who have recovered and are now immune.
* Deceased (D): Individuals who have died from the disease.

The state of the population at any given time t (in days) is represented by a 4-vector x_t, which gives the proportion of the population in each state:
$$ x_t = \begin{bmatrix} S_t \ I_t \ R_t \ D_t \end{bmatrix} $$

The progression of the epidemic from one day to the next is governed by the following daily transition rules:

Among the susceptible population:
- 5% acquire the disease (become Infected).
- 95% remain Susceptible.

Among the infected population:
- 1% dies (become Deceased).
- 10% recover with immunity (become Recovered).
- 4% recover without immunity (become Susceptible).
- 85% remain Infected.

100% of immune (Recovered) and Deceased people remain in their respective states.

This process can be modeled as a linear system:
$$ x_{t+1} = A x_t $$
where A is the transition matrix that describes the daily changes.

- **[1 Marks]** Based on the transition rules provided above, manually determine the 4x4 transition matrix A for the linear system $x_{t+1} = A x_t$.
- **[1 Marks]** Start with an initial state vector of $x_1 = [1,0,0,0]^T$. This represents a scenario where, on day 1, the entire population is susceptible. Iteratively calculate the state vector for each subsequent day (from t=2 to t=200). Store the results for each of the four population groups for every day. Create a single plot that visualizes the simulation results. The x-axis should represent Time t (in days), from 1 to 200. The y-axis should represent the Proportion of the population, $x_t$, from 0 to 1. Plot the trends for all four groups (Susceptible, Infected, Recovered, and Deceased) on the same graph.

<!-- ![imgsrc](sample_images/Epidemic_model.png) -->

<div style="display: flex; justify-content: center; align-items: center;">
    <img src="sample_images/Epidemic_model.png" width="80%"/>
</div>
