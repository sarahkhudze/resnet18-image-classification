ResNet-18 on MNIST Dataset
This repository contains the implementation of a ResNet-18 model applied to the MNIST dataset using PyTorch in a Jupyter Notebook.

Overview
The project implements ResNet-18, a deep residual network, to classify handwritten digits from the MNIST dataset. It demonstrates the use of convolutional neural networks (CNNs) for image classification and includes the following key components:

Data Preprocessing: The MNIST dataset is loaded and transformed into tensors for training.
Model Architecture: A custom ResNet-18 model is implemented using PyTorch’s neural network modules.
Training: The model is trained using the Adam optimizer and cross-entropy loss function.
Evaluation: The model's performance is evaluated on both the training and test datasets.
Visualization: The notebook visualizes sample images and model predictions.
Notebook Description
1. Data Loading and Preprocessing
The MNIST dataset is automatically loaded and transformed for training. This includes data augmentation, scaling, and converting the images to tensors. The dataset is split into training and test sets, with DataLoaders used for efficient batching.

2. Model Architecture
The ResNet-18 architecture is implemented with the following key components:

Convolutional layers
Batch normalization
ReLU activations
Residual connections that allow the network to learn more complex features without degrading performance
3. Training Loop
The model is trained using the Adam optimizer and the cross-entropy loss function. The training loop includes:

Forward propagation
Loss computation
Backpropagation
Weight updates
4. Accuracy Calculation
Accuracy is computed for both training and testing datasets after each epoch to evaluate model performance.

5. Visualization
The notebook visualizes the following:

Sample images from the dataset along with their true labels
Predicted labels for test images
Accuracy plots after training
Key Functions
The following are key functions used in the notebook:

conv3x3: Defines a 3x3 convolutional layer with padding.
BasicBlock: Implements the basic block used in the ResNet architecture.
ResNet: Defines the ResNet model, which consists of several residual blocks.
compute_accuracy: Computes and returns the accuracy of the model on a given dataset.
train_model: Runs the training loop, including forward and backward passes, and updates the model parameters.
visualize_results: Displays sample images and their predicted labels.
Training and Results
The notebook trains the ResNet-18 model on the MNIST dataset for a specified number of epochs. The final accuracy for both the training and testing datasets is displayed, and sample predictions are visualized.

Sample Output:
Epoch: 001/010 | Batch 0050/468 | Cost: 0.3021
Epoch: 002/010 | Train: 98.72%
Test accuracy: 98.53%
Visualization:
The notebook visualizes the predicted labels for test images. Below is a sample output:


Conclusion
This notebook demonstrates the application of ResNet-18 to the MNIST dataset, showcasing how deep learning models can achieve high accuracy on image classification tasks. It serves as an example of how to implement and train a deep neural network for image classification using PyTorch.
