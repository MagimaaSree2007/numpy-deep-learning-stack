# numpy-deep-learning-stack
# Deep Learning From Scratch Using NumPy

## Overview

This project implements a deep learning system from first principles using Python and NumPy, without relying on deep learning frameworks such as PyTorch, TensorFlow, or JAX.

The main objective is to understand how neural networks work internally by implementing automatic differentiation, backpropagation, optimization algorithms, and neural network layers manually.

The project uses the Fashion-MNIST dataset to train and evaluate a Multi-Layer Perceptron (MLP) and a Convolutional Neural Network (CNN).

## Objectives

- Implement a tensor system with reverse-mode automatic differentiation.
- Support mathematical operations such as addition, matrix multiplication, ReLU, broadcasting, and softmax cross-entropy.
- Implement backpropagation and verify gradients using numerical gradient checking.
- Train an MLP using custom optimization algorithms.
- Build a CNN with convolution, max pooling, batch normalization, and dropout.
- Evaluate model performance using training, validation, and test datasets.
- Analyze model performance and identify possible limitations.

## Technologies Used

- **Python** — Main programming language
- **NumPy** — Numerical computation and tensor operations
- **Matplotlib** — Training and evaluation plots
- **Google Colab** — Development and execution environment
- **Fashion-MNIST** — Image classification dataset

## Dataset

Fashion-MNIST contains grayscale images of clothing items belonging to 10 categories. Each image has a resolution of 28 × 28 pixels.

The dataset includes:
- 60,000 training images
- 10,000 test images

The training data is further divided into training and validation subsets. Images are normalized before being passed to the neural network.

## Project Implementation

### 1. Automatic Differentiation

A custom Tensor class is implemented using NumPy. It tracks operations and their computational graph to calculate gradients through reverse-mode automatic differentiation.

Supported operations include:
- Addition and broadcasting
- Matrix multiplication
- ReLU activation
- Reshaping and reduction operations
- Softmax cross-entropy loss
- Backpropagation using `.backward()`

### 2. Multi-Layer Perceptron

The MLP is trained on Fashion-MNIST using manually implemented neural network operations and optimization algorithms.

The training process tracks loss and accuracy to monitor learning performance.

### 3. Convolutional Neural Network

The CNN includes:
- Conv2D implemented using NumPy operations and image patches
- MaxPool2D for spatial downsampling
- Batch normalization
- ReLU activation
- Dropout for regularization
- Fully connected layers for classification

The CNN is trained using a custom optimization pipeline and evaluated on unseen test data.

## CNN Results

The best observed results are:

| Metric | Result |
|---|---:|
| Best validation accuracy | 91.68% |
| Final test accuracy | 90.85% |
| Test loss | 0.2707 |
| Target test accuracy | 91.00% |

The CNN achieved 91.68% validation accuracy and 90.85% test accuracy. The test result is slightly below the target of 91%.

The difference between validation and test accuracy indicates that performance on unseen data can differ from performance on the validation set.

## Gradient Verification

Numerical gradient checking is used to compare analytical gradients from backpropagation with numerical approximations. The target relative error is below `1e-5`.

This helps verify that the manually implemented backward operations are mathematically consistent.

## Training and Evaluation

The project records training loss, validation loss, training accuracy, and validation accuracy. These metrics can be plotted to analyze convergence and identify potential overfitting.

A smaller batch size is used during CNN evaluation and training when necessary to reduce RAM usage in Google Colab.

## How to Run

1. Open the notebook in Google Colab.
2. Run the setup and import cells.
3. Load and preprocess the Fashion-MNIST dataset.
4. Execute the Tensor and automatic differentiation implementation.
5. Run the gradient verification cells.
6. Train and evaluate the MLP.
7. Run the CNN layer implementations and their tests.
8. Train the CNN and evaluate its performance.
9. Review the saved model checkpoint and training plots.

Run the notebook cells in order because later cells depend on classes, functions, and variables created earlier.

## Limitations

- The CNN test accuracy is 90.85%, slightly below the 91% target.
- Training can require considerable RAM because the CNN uses NumPy-based convolution and backpropagation.
- Training time depends on the available runtime resources.
- Additional optimization and experimentation may improve generalization performance.

## Future Improvements

- Improve CNN generalization and test accuracy.
- Optimize memory usage and training speed.
- Complete and evaluate post-training INT8 quantization and integer-only inference.
- Investigate robustness under image perturbations.
- Explore knowledge distillation and learning-rate scheduling.

Advanced features should be considered complete only after their implementations and results have been verified.

## Conclusion

This project demonstrates how core deep learning components can be implemented from scratch using NumPy. By building automatic differentiation, neural network layers, and optimization procedures manually, the project provides practical insight into backpropagation and image classification without relying on a high-level deep learning framework.

