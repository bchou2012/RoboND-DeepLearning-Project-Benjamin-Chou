Udacity Deep Learning Project
=

This project covers the theory and techniques in training a neural network to identify images of a target being followed by a quadrotor. 

## Segmentation Network Theory

**Logistic Regression**

Given two or more groups of data within a data set, a boundary can be drawn separating the data groups. This boundary is defined by a function whose parameters can be determined through logistic regression. The goal of logistic regression function is to predict the probability of a data point being within a boundary defined by an equation. The parameters are tuned through iteration with a training data set, with the rate of tuning weighted by a learning rate. 

<img src="https://github.com/bchou2012/RoboND-DeepLearning-Project-Benjamin-Chou/blob/master/writeup_media/Statistical%20Fitting.jpg?raw=true" alt="Examples of Boundary Fitting" width="480px">

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

<img src="https://github.com/vdumoulin/conv_arithmetic/blob/master/gif/padding_strides.gif?raw=true" alt="animated convolution example 2" width="320px">

*Image Source: https://github.com/vdumoulin/conv_arithmetic/*

The above example shows a 5x5 matrix with a 3x3 convolution run on it with a stride of 2 and zero padding along the edges.

**Neurons and Rectifier Linear Unit (ReLU)**

While the images are downsized through convolutions, hidden layers are also created. These layers contain neurons, which take inputs and produce an output similar to logic gates. For the purposes of this project the neurons use two activations, rectilinear and softmax. 

Rectilinear produces a nonlinear output, zero below a threshold, and a linear output above the threshold. Rectilinear units (ReLU) are used in the convolution blocks of the project. 

<img src="https://cdn.tinymind.com/static/img/learn/relu.png" alt="ReLu Example" width="320px">

*Image source: https://www.tinymind.com/learn/terms/relu*

Softmax returns a vector that represents a probability distribution of the inputs.  This is used at the end of the fully convolutional network of the project to return the constructed probability output. 

<img src="https://cdn-images-1.medium.com/max/1000/0*2r10e7gw1jzOsHhC.png" alt="Softmax Example" width="320px">

*Image source: https://towardsdatascience.com/deep-learning-concepts-part-1-ea0b14b234c8* 

**Fully Convolutional Neural Network**

<img src="https://github.com/bchou2012/RoboND-DeepLearning-Project-Benjamin-Chou/blob/master/writeup_media/Fully%20Convolutional%20Network.jpg?raw=true" alt="Sample FCN Diagram" width="640px">

The input is an image, broken down into three layers which represent the image's R, G, and B value at each pixel. The input undergoes any number of convolutions with smaller and smaller convolution matrix sizes but greater depth. After the last convolution the filters undergo a 1x1 convolution to reduce the number of parameters and decrease processing time. 

The classifcations are then built back up into an output prediction image using deconvolutional layers, which builds up larger images from smaller images using bilinear upsampling, which predicts what the missing data values would be in upscaling.  

Skip connections are added, which connects convolutional layers to deconvolutional layers. This helps with the possible data loss and artifacts created through bilinear upsampling by providing the data of same filter level before the downsampling/upsampling process.  
 
## Collecting Training Data

Due to time constraints the GitHub provided data set was used to train the network. While exploration of the environment and pathing was set in QuadSim, a satisfactory and thorough data set was unable to be recorded. 

<img src="https://github.com/bchou2012/RoboND-DeepLearning-Project-Benjamin-Chou/blob/master/docs/misc/sim_screenshot.png?raw=true" alt="Quadsim in action" width="320px">


## Implement the Segmentation Network

**Network Construction**

The provided `model_training.ipynb` file provided preconstructed functions for separable and regular convolutions, and bilinear upsampling, defined as `separable_conv2d_batchnorm()`, `conv2d_batchnorm()` and `bilinear_upsample()`. The separable convolution runs the input filter through a 1x1 convolution to reduce dimensionality as discussed previously. 

*Encoder Block*
The encoder block runs an input layer through the `separable_conv2d_batchnorm()`, along with filter size and stride parameters. The input layer undergoes convolution and a hidden layer with ReLU activation, which is then returned. This function is defined as `encoder_block()`

*Decoder Block*
The decoder block upsizes the input layer with the `bilinear_upsample()`. Logic is added to allow for selection of a skip layer to be concatenated with the upsampled input, then run through the `seperable_conv2d_batchnorm()`. This function is defined as `decoder_block()`

*Fully Convolutional Network*

Constructing the Fully Convolutional Network is as follows:

..* A number of encoder blocks are added, with each depth increasing the number of filters.
..*

**Hyperparameters**

*Learning Rate*

From logistic regression theory, the learning rate is the rate of change of correction of the training functions. A large learning rate approaches the optimal prediction coefficients faster, but also increases the overshoot as the coefficients approach the optimum. A smaller learning rate means the function changes more slowly, but decreases overshoot and increases accuracy. 

<img src="https://qph.fs.quoracdn.net/main-qimg-ed4a3867ca90b95b33b95f1b89d8335c.webp" alt="Learning Rate example" width="320px">

*Image Source: https://www.quora.com/What-is-the-meaning-of-changing-the-LR-learning-rate-in-neural-networks*

In conjunction with number of epochs and validation steps, a small learning rate may also cause overfitting of the functions, which means the function is too attuned to the testing data and may not classify outside data, especially if outside data has features not included in the testing data. 

*Batch Size, Steps, and Epochs*

These three hyperparameters have effects on each other. Batch size is how many data sets (in this case images) are processed per pass. Steps per Epoch is how many batches are run per epoch. To process all the images in a single epoch, the product of the batch size and the steps would have to be at least equal to the number of images. Epochs is the number of times the whole data set is processed.

A large batch size means more data sets are processed before the function corrects its parameters, but is resource-intensive. A smaller batch size uses less resources, but correcting the function based on small batch sizes means a greater chance of error. 

A large epoch size is useful for small learning rates, as small learning rates propagate parameter corrections slowly. If an epoch size is too large the classifier may overcorrect and end up moving away from the desired results. A small epoch size reduces run time and the possibility over overcorrection, but depending on the learning rate may not reach the global minimum error. 

<img src="" alt="" width="640px">

*Validation Steps*

Validation steps is the number of validation sets checked against the training sets. A large validation set overtrains the training set against the validation data which may leave the output results too fitted against the validation data, and provide less accuracy against "real" data inputs. 


**Prediction Process**

After the FCN is trained, the predictions are compared to the validation image set. 

The three prediction parameters are:

..*
..*
..*
## Results


*Bilinear Upsampling and Strides*

*Input Resolution*

Runs with using the full image size of 256 H x W provided the final passing results. Downsizing to 160 H x W as suggested by the project code means scaling the base image down by almost 40%. Combined with the low batch size this co

Scaling down the image size allows for smaller inputs and faster processing. For the context of this project the scaling down to 160 H x W consistently proved to enough detail loss to consistently miss the 40% passing rate, in conjunction with other implementation limitations. 

*Batch size vs Filter Depth*


<img src="" alt="" width="640px">
<img src="" alt="" width="640px">
<img src="" alt="" width="640px">
<img src="" alt="" width="640px">
<img src="" alt="" width="640px">
<img src="" alt="" width="640px">