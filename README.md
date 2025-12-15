Neural Networks & Latent Space Classification (From Scratch)
Project Overview

This project involves implementing Neural Networks from scratch using NumPy, focusing on understanding core concepts such as:

Forward propagation

Backpropagation

Optimization using gradient descent

Key Components:

Part 1: XOR Problem (Implement a neural network from scratch to solve the XOR problem)

Part 2: Autoencoder for MNIST (Build an autoencoder to compress MNIST digits and visualize reconstructions)

Part 3: Latent Space Classification (Train an SVM in the latent space of the autoencoder and evaluate performance)

Part 4: Baseline Comparison (Compare implementations in TensorFlow/Keras and the from-scratch version)

Installation

Clone the repository and install the required dependencies:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt


Make sure you have Jupyter Notebook installed, or install it via:

pip install notebook

Sections and Key Objectives
Section 1: XOR Problem (From Scratch)

Objective: Build a neural network to solve the XOR classification problem using backpropagation and SGD optimization.

Key Points: Gradient checking, correct implementation of forward/backward pass, training loop from scratch.

Section 2: Autoencoder for MNIST (From Scratch)

Objective: Train an autoencoder on MNIST images, compressing the data into a latent space representation.

Key Points: Understanding how autoencoders work, training with mini-batch SGD, and visualizing reconstructions.

Section 3: Latent Space Classification

Objective: Use the compressed latent features from the autoencoder and apply SVM classification.

Key Points:

Part 3A: Train an SVM using sklearn on the latent space features and evaluate performance.

Part 3B: Implement a linear SVM from scratch using SGD and compare results with sklearn's implementation.

Visualization of confusion matrices and classification metrics.

Section 4: Baseline Comparison

Objective: Implement the same architectures in TensorFlow/Keras to compare the ease of implementation, training time, and final performance.

Key Points: Compare results of from-scratch SVM with TensorFlow/Keras implementations.

Features

XOR Problem: A fully working neural network with manual backpropagation and SGD optimization.

Autoencoder: Successfully trains an autoencoder to compress and reconstruct MNIST images.

Latent Space Classification: Uses the learned features from the autoencoder for downstream classification using SVM.

Baseline Comparison: Compares the from-scratch neural network models with TensorFlow/Keras implementations to evaluate performance and ease of use.

How to Run the Project

Clone the repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


Open the Jupyter notebook:

jupyter notebook main_notebook.ipynb


Run the notebook from top to bottom (important, especially after runtime restarts).

Files in this Project

network.py: Network structure (forward/backward pass, architecture)

layers.py: Layer definitions (Dense, etc.)

activations.py: Activation functions (ReLU, Sigmoid)

losses.py: Loss functions (MSE, BCE)

optimizer.py: Optimization algorithms (SGD, Adam)

SVM.py: Linear SVM from scratch using SGD and hinge loss

main_notebook.ipynb: Jupyter notebook containing the full project implementation and results

Results and Comparison
Aspect	From Scratch	TensorFlow/Keras
Ease of Implementation	Difficult	Very Easy
Training Time	Slower	Faster
Final Reconstruction Loss	Higher	Lower
Code Length	Longer	Shorter
Summary:

The from-scratch implementation provides deeper insight into optimization and neural network mechanics, but the TensorFlow/Keras version is significantly easier to implement and faster to train.

License

This project is licensed under the MIT License - see the LICENSE
 file for details.

Acknowledgements

MNIST dataset: This project uses the MNIST dataset for digit classification and autoencoder reconstruction.

TensorFlow and Keras: Used for baseline comparison in Part 4.

Conclusion

This project demonstrates the process of building machine learning models from scratch and highlights the importance of feature extraction through autoencoders. The comparison with TensorFlow/Keras shows how modern frameworks simplify implementation but also hides much of the underlying complexity.
