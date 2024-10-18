# Application-of-Neural-Networks
This project aims to explore the performance of a CNN-based model in classifying CIFAR-10 im-
ages

## Installation
To run this notebook, you are required to have the following dependencies installed:

    - `torch`
    - `torchvision`
    
These can be installed using pip:
    `pip install torch torchvision`

## Dataset
The dataset is automatically downloaded using `torchvision` when running the notebook.
It is split into training and testing datasets, with data loaders for each one respectively.
The dataset contains 10 classes:
    
    - Bird
    - Cat
    - Deer
    - Dog
    - Horse
    - Airplane
    - Automobile
    - Frog
    - Ship
    - Truck

## Training
The model is trained using the following configurations:

    - Optimizer: Adam Optimizer with a learning rate of 0.001
    - Schedular: Cosine Annealing Schedular to dynamically adjust the learning rate during training
    - Loss function: Cross-Entropy Loss
    - Number of epochs: 50
    - Batch size: 32

## Usage
Extract and run the notebook using Google Colab, Jupyter Notebook or Jupyter Lab.
The model weights are saved to `trained_model.pth` after training.

## Results
After training the model for 50 epochs, it achieved an accuracy of approximately 81.77% on the CIFAR-10 test dataset.
