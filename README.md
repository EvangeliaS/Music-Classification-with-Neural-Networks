This project focuses on music genre prediction using neural networks. The main objective is to classify 1-second music samples into four genres: classical, pop, rock, and blues. Two different audio data representations, MFCCs (Mel-frequency cepstral coefficients) and mel-spectrograms, are utilized.

MFCCs capture spectral characteristics transformed based on the mel scale, approximating human auditory perception. Each music sample generates a 26-dimensional feature vector by computing the mean and standard deviation for each of the 13 coefficients across 20 time frames.

Mel-spectrograms, on the other hand, provide a 21x128 matrix representing the time-frequency evolution of audio spectrum when applying the mel scale to the spectrogram.

The dataset is divided into training, validation, and test sets, comprising 3200, 800, and 1376 samples, respectively. PyTorch is used to preprocess the data, build and train the neural network model.

The implementation includes data loading, model definition (a fully connected neural network), training and evaluation processes, and model selection based on validation performance.

**Step 2: Define a Fully Connected Neural Network**
- Create a class for a fully connected neural network with four layers having 26, 128, 32, and 4 neurons, respectively.

**Step 3: Define the Training Process**
- Define a function responsible for training the neural network.
- Train the model using stochastic gradient descent as the optimizer, a learning rate of 0.002, cross-entropy loss, and 30 epochs.
- Evaluate the model's performance on the test set using various metrics.

**Step 4: Define the Evaluation Process**
- Define a function for evaluating the model on a dataset, calculating loss, F1 score (macro-averaged), accuracy, and the confusion matrix.

**Step 5: Train the Network with GPU**
- Repeat the training process, but this time, move the data and the initialized network to the GPU for accelerated training.
- Compare the execution times on GPU and CPU.

**Step 6: Model Selection**
- During training, save snapshots of the model.
- Choose the model with the best F1 score on the validation set.
- Evaluate the selected model on the test set.


```markdown
# Music Genre Classification with Neural Networks

This project aims to predict music genres from 1-second music samples using neural networks. Two representations of audio data, MFCCs and mel-spectrograms, are used to train and evaluate the models.

## Project Structure

- `data/`: Contains the data files (X.npy, labels.npy) and datasets.
- `models/`: Contains the Python code for the neural network model.
- `utils/`: Contains utility functions for data loading, preprocessing, and evaluation.
- `train.ipynb`: Jupyter notebook with the code for training and evaluating the models.
- `evaluate.ipynb`: Jupyter notebook for evaluating the trained models on the test set.

## Results

The project achieves a certain level of accuracy and F1 score for music genre classification. Detailed results and insights can be found in the notebooks.
