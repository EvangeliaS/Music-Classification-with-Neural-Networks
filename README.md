# Music Genre Classification with Neural Networks

Evangelia Steiropoulou

## Introduction

This project focuses on music genre classification using both fully connected and convolutional neural networks. The main objective is to classify 1-second music samples into four genres: classical, pop, rock, and blues. Two different audio data representations, MFCCs (Mel-frequency cepstral coefficients) and mel-spectrograms, are utilized.

MFCCs capture spectral characteristics transformed based on the mel scale, approximating human auditory perception. Each music sample generates a 26-dimensional feature vector by computing the mean and standard deviation for each of the 13 coefficients across 20 time frames.

Mel-spectrograms, on the other hand, provide a 21x128 matrix representing the time-frequency evolution of audio spectrum when applying the mel scale to the spectrogram.

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

## Fully Connected Neural Network

In the fully connected neural network section, you can provide details about the architecture of the model, including the number of layers, activation functions, and any specific design choices you made. Additionally, you can explain the training process, such as the optimizer used, learning rate, and the number of epochs.

## Convolutional Neural Network

Here, you can describe the architecture of the convolutional neural network (CNN) used in the project. Include information about the number of convolutional layers, max-pooling layers, and any other architectural details that are relevant. Don't forget to mention the purpose of using a CNN for this task.

## Mel-spectrograms

In this section, we discuss the utilization of mel-spectrograms as an alternative audio data representation. Explain the process of generating mel-spectrograms from the audio samples and how they are used as input data for the neural network models.

## Optimization Algorithms

We have tested various optimization algorithms, and you can find the results in the `optimization_algorithms.py` script. 
$Adagrad$ and $Adamax$ have the best accuracy and the best f1 score as well, as I noticed after several test runs. For this specific runtime, $Adagrad$ has the best scores on both accuracy and f1-score.


## Results

We tested several optimization algorithms and found that Adagrad performed the best in terms of both accuracy and F1 score. You can find the detailed results in the project.

