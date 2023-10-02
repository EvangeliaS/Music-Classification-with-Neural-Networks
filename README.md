# Music Genre Classification with Neural Networks

Evangelia Steiropoulou

## Introduction

This project focuses on music genre classification using both fully connected and convolutional neural networks. The main objective is to classify 1-second music samples into four genres: classical, pop, rock, and blues. Two different audio data representations, MFCCs (Mel-frequency cepstral coefficients) and mel-spectrograms, are utilized.

## Table of Contents

- [Dataset](#dataset)
- [MFCCs (Mel-frequency cepstral coefficients)](#MFCCs (Mel-frequency-cepstral-coefficients))
  - [Fully Connected Neural Network](#fully-connected-neural-network)
- [Mel-spectrograms](#mel-spectrograms)
  - [Convolutional Neural Network](#convolutional-neural-network)
  - [Optimization Algorithms](#optimization-algorithms)
  - [Results](#results)

## Dataset

The dataset used in this project is composed of audio samples from four different music genres: classical, pop, rock, and blues. It is divided into training, validation, and test sets, comprising 3200, 800, and 1376 samples, respectively. Unfortunately, the dataset is not publicly available at the moment, but you can easily replace it with your own music dataset. PyTorch is used to preprocess the data, build and train the neural network model.

## MFCCs (Mel-frequency cepstral coefficients)

MFCCs capture spectral characteristics transformed based on the mel scale, approximating human auditory perception. Each music sample generates a 26-dimensional feature vector by computing the mean and standard deviation for each of the 13 coefficients across 20 time frames.

### Fully Connected Neural Network

The fully connected neural network used in this project is designed with four layers, each specifying the number of neurons in that layer. The architecture of the neural network is as follows:

- Input Layer: 26 neurons
  - The input layer has 26 neurons, corresponding to the dimensionality of the input data.

- Hidden Layer 1: 128 neurons
  - The first hidden layer consists of 128 neurons, which apply a Rectified Linear Unit (ReLU) activation function.

- Hidden Layer 2: 32 neurons
  - The second hidden layer consists of 32 neurons, also utilizing the ReLU activation function.

- Output Layer: 4 neurons
  - The output layer contains 4 neurons, representing the number of classes for music genre classification.

The neural network takes 26-dimensional input data and passes it through these layers to make predictions for the music genre. It is trained using stochastic gradient descent (SGD) with a learning rate of 0.002 and the CrossEntropyLoss as the loss function. 
The model is trained on CPU and GPU, and the performance is evaluated.
Here are the results:

- **CPU Execution Time**: 9.05 seconds
- **GPU Execution Time**: 13.15 seconds

### Accuracy Comparison

- **CPU Accuracy**: 62.14%
- **GPU Accuracy**: 56.40%

### F1-Score Comparison

- **CPU F1-Score**: 0.611
- **GPU F1-Score**: 0.554

As observed, there is a trade-off between execution time and model performance when switching from CPU to GPU. While the GPU execution is slightly slower in this case, it's important to note that it offers a speed advantage for more complex models or larger datasets. Additionally, the CPU setup achieved a slightly higher accuracy and F1-score compared to the GPU. The choice between CPU and GPU may depend on the specific use case and hardware availability.

## Mel-spectrograms

Mel-spectrograms, provide a 21x128 matrix representing the time-frequency evolution of audio spectrum when applying the mel scale to the spectrogram.

### Convolutional Neural Network

Here, you can describe the architecture of the convolutional neural network (CNN) used in the project. Include information about the number of convolutional layers, max-pooling layers, and any other architectural details that are relevant. Don't forget to mention the purpose of using a CNN for this task.


## Optimization Algorithms

We have tested various optimization algorithms, and you can find the results in the `optimization_algorithms.py` script. 
$Adagrad$ and $Adamax$ have the best accuracy and the best f1 score as well, as I noticed after several test runs. For this specific runtime, $Adagrad$ has the best scores on both accuracy and f1-score.


## Results

We tested several optimization algorithms and found that Adagrad performed the best in terms of both accuracy and F1 score. You can find the detailed results in the project.

