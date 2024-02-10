# Assignment 2

Total marks: 

For all the questions given below, create `assignment_q<question-number>_subjective_answers.md` and write your observations.

## Questions
1. Implement momentum [2]
2. `np.linalg.solve` v/s `np.linag.inv` reference lecture on multi-colinearity [1]
3. Sklearn's linear regression code does not suffer due to multi-colinearity. Why? Dive deep into sklearn's linear regression code and explain why [1]
4. Begin by exploring the [instructor's notebook](https://github.com/nipunbatra/ml-teaching/blob/master/notebooks/siren.ipynb) that introduces the application of Random Fourier Features (RFF) for image reconstruction. Demonstrate the following applications using the cropped image from the notebook:
    - Superresolution: perform superresolution on the image shown in notebook to enhance its resolution by factor 2. **[2 Marks]**
    - Completing Image with Random Missing Data: Apply RFF to complete the image with 10%, 20%, and so on up to 90% of its data missing randomly. Randomly remove portions of the data, train the model on the remaining data, and predict on the entire image. Display the reconstructed images for each missing data percentage. **[1 Mark]**
    - Image Inpainting: Take out a rectangular patch from the image. Utilize RFF to train a linear model using the remaining image data. Reconstruct the entire image using the trained model and highlight the effectiveness of the inpainting process. **[1 Mark]**
5. Use instructor's notebook on matrix factorisation
    - use above image from Q4 and complete the rectangular missing patch [1]
    - write a function using https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html and use alternating least squares instead of gradient descent [1]
6. Solve the activity recognition problem using RF and linear regression and compare the results to DT. [2]
