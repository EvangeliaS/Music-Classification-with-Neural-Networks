# Music Genre Classification with Neural Networks

Evangelia Steiropoulou

## Introduction

This project focuses on music genre classification using both fully connected and convolutional neural networks. The main objective is to classify 1-second music samples into four genres: classical, pop, rock, and blues. Two different audio data representations, MFCCs (Mel-frequency cepstral coefficients) and mel-spectrograms, are utilized.

MFCCs capture spectral characteristics transformed based on the mel scale, approximating human auditory perception. Each music sample generates a 26-dimensional feature vector by computing the mean and standard deviation for each of the 13 coefficients across 20 time frames.

Mel-spectrograms, on the other hand, provide a 21x128 matrix representing the time-frequency evolution of audio spectrum when applying the mel scale to the spectrogram.

## Table of Contents

- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Fully Connected Neural Network](#fully-connected-neural-network)
  - [Convolutional Neural Network](#convolutional-neural-network)
  - [Optimization Algorithms](#optimization-algorithms)
- [Results](#results)

## Dataset

The [dataset]() is divided into training, validation, and test sets, comprising 3200, 800, and 1376 samples, respectively. PyTorch is used to preprocess the data, build and train the neural network model.

### Fully Connected Neural Network

To train and evaluate a fully connected neural network, follow these steps:

1. Run `fully_connected_nn.py` to train the model.
2. Evaluate the model by running `evaluate_fully_connected_nn.py`.

### Convolutional Neural Network

To train and evaluate a convolutional neural network, follow these steps:

1. Run `cnn.py` to train the model.
2. Evaluate the model by running `evaluate_cnn.py`.

### Optimization Algorithms

We have tested various optimization algorithms, and you can find the results in the `optimization_algorithms.py` script. To compare different optimizers, run this script.

## Results

We tested several optimization algorithms and found that Adagrad performed the best in terms of both accuracy and F1 score. You can find the detailed results in the project.
