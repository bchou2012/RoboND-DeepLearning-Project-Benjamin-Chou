Udacity Deep Learning Project
=

This project covers the theory and techniques in training a neural network to identify images of a target being followed by a quadrotor. 

## Segmentation Network Theory

**Logistic Regression**

Given two or more groups of data within a data set, a boundary can be drawn separating the data groups. This boundary is defined by a function whose parameters can be determined through logistic regression. The goal of logistic regression function is to predict the probability of a data point being within a boundary defined by an equation. The parameters are tuned through iteration with a training data set, with the rate of tuning weighted by a learning rate. 

<img src="" alt="Examples of Boundary Fitting" width="480px">

In the scope of this project, logistic regression is used to create predictive functions that define image features at all scales of the image, from points and simple lines to geometric shapes to distinct objects. 

**Image Recognition Theory**

Training a system to recognize full scale images as is would be computationally taxing and would require a prohibitively large data set. Breaking down each image in progressive sections would allow for training for recognition of simpler features at smaller scales, and then building those models back up. 

**Convolutions**

Downsizing images make use of convolution. A convolution matrix takes a pixel and a number of its local neighbors  in an image, dependent on the convolution size, then creates a weighted value, with the resulting output a smaller matrix. 

<img src="http://deeplearning.net/software/theano/_images/no_padding_no_strides.gif" alt="animated convolution example" width="320px">

*Image source: http://deeplearning.net/*
 
In the above example, a 4x4 matrix has a 3x3 convolution run on it, with an output size of 2x2. It has a stride of 1, meaning it moves 1 value on the input each time it runs the convolution. 

For larger matrices larger strides can be used, which reduces output size at the possible cost of losing output fidelity, as some positions of the matrix will be skipped during some of the convolutions. 

The above example also shows no padding, as the convolution matrix passes over edge and corner positions far less than interior positions. Zero padding adds zeros around the perimeter of the matrix, allowing the edge positions to be evaluated as normally as non-edge positions. 

<img src="https://github.com/vdumoulin/conv_arithmetic/blob/master/gif/padding_strides.gif?raw=true" alt="animated convolution example 2" width="329px">

*Image Source: https://github.com/vdumoulin/conv_arithmetic/*

The above example shows a 5x5 matrix with a 3x3 convolution run on it with a stride of 2 and zero padding along the edges.

**Neurons and Rectifier Linear Unit (ReLU)**

While the images are downsized through convolutions, hidden layers are also created. These layers contain neurons, which take inputs and produce an output similar to logic gates. For the purposes of this project the neurons use two activations, rectilinear and softmax. 

Rectilinear produces a nonlinear output, zero below a threshold, and a linear output above the threshold. Rectilinear units (ReLU) are used in the convolution blocks of the project. Softmax returns a vector that represents a probability distribution. This is used at the end of the fully convolutional network of the project to return the constructed probability output. 


**Fully Convolutional Neural Network**

<img src="https://www.mathworks.com/content/mathworks/www/en/solutions/deep-learning/convolutional-neural-network/jcr:content/mainParsys/band_copy_copy_14735_1026954091/mainParsys/columns_1606542234_c/2/image.adapt.full.high.jpg/1530247928116.jpg" alt="Convolutional Network Structure" width="640px">

*Image Source: https://www.mathworks.com/solutions/deep-learning/convolutional-neural-network.html*

The input is an image, broken down into three layers which represent the image's R, G, and B value at each pixel. The input undergoes any number of convolutions with smaller and smaller convolution matrix sizes but greater depth. After the last convolution the data undergoes a 1x1 convolution to reduce dimensionality and decrease processing time. 

The predictions are then built back up into an output prediction image using deconvolutional layers, which builds up larger images from smaller images using bilinear upsampling. 

For a fully convolutional neural network, skip connections are added, which connects convolutional layers to deconvolutional layers. This helps with the possible data loss and artifacts created through bilinear upsampling. 
 

*1x1 Convolution*






## Collecting Training Data
Due to time constraints the GitHub provided data set was used to train the network. While exploration of the environment and pathing was set in QuadSim, a satisfactory and thorough data set was unable to be recorded. 

## Implement the Segmentation Network

The provided `model_training.ipynb` file provided preconstructed functions for separable and regular convolutions, defined as `separable_conv2d_batchnorm()` and  `conv2d_batchnorm()`

Using those

*Encoder Block*
The encoder block runs an input through the `separable_conv2d_batchnorm()`

*Decoder Block*

**Hyperparameters**

*Learning Rate*

From logistic regression theory, the learning rate is the rate of change of correction of the training functions. A large learning rate approaches the optimal prediction coefficients faster, but also increases the overshoot as the coefficients approach the optimum. A smaller learning rate means the function changes more slowly, but decreases overshoot and increases accuracy. 

<img src="https://qph.fs.quoracdn.net/main-qimg-ed4a3867ca90b95b33b95f1b89d8335c.webp" alt="Learning Rate example" width="320px">

*Image Source: https://www.quora.com/What-is-the-meaning-of-changing-the-LR-learning-rate-in-neural-networks*

In conjunction with number of epochs and validation steps, a small learning rate may also cause overfitting of the functions, which means the function is too attuned to the testing data and may not classify outside data, especially if outside data has features not included in the testing data. 

*Batch Size, Steps, and Epochs*

These three hyperparameters have effects on each other. Batch size is how many data sets (in this case images) are processed per pass. Steps per Epoch is how many batches are run per epoch. To process all the images in a single epoch, the product of the batch size and the steps would have to be at least equal to the number of images. Epochs is the number of times the whole data set is processed.

A large batch size means more data sets are processed before the function corrects its parameters, but is resource-intensive. A smaller batch size uses less resources, but correcting the function based on small sample sizes means a greater chance of error. 

A large epoch size m. A small epoch size

*Validation Steps*

Validation steps is the number 




<img src="" alt="" width="640px">
<img src="" alt="" width="640px">
<img src="" alt="" width="640px">
<img src="" alt="" width="640px">
<img src="" alt="" width="640px">
<img src="" alt="" width="640px">