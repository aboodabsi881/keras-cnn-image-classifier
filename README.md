# CIFAR-10 Image Classification using CNNs

A deep learning project implementing a Convolutional Neural Network (CNN) in TensorFlow and Keras to classify images into 10 distinct categories. The model is trained and evaluated using the classic CIFAR-10 computer vision dataset.

---

## 📌 Project Overview
The objective of this project is to build an end-to-end image classification pipeline. It demonstrates how to load and preprocess pixel-level image data, scale feature maps, configure a sequential CNN architecture with inter-leaved convolutional and pooling layers, and train the model using efficient optimization functions.

## 📊 Dataset: CIFAR-10
The model uses the CIFAR-10 dataset, which consists of 60,000 32x32 color images across 10 classes (6,000 images per class). The dataset is split into 50,000 training images and 10,000 test images. 

The target categories include:
* Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.

---

## 🛠️ Model Architecture & Workflow

1. **Data Normalization:** Image pixel values are scaled down from their original integer range `[0, 255]` to a floating-point range `[0.0, 1.0]` to ensure stable and faster gradient descent convergence during backpropagation.
2. **Feature Extraction (Convolutional Base):**
   * **Conv2D Layer 1:** 32 filters (3x3 kernel), ReLU activation. Processes raw input shapes of `(32, 32, 3)`.
   * **MaxPooling2D Layer 1:** Downsamples feature maps using a 2x2 window.
   * **Conv2D Layer 2:** 64 filters (3x3 kernel), ReLU activation.
   * **MaxPooling2D Layer 2:** 2x2 max pooling.
   * **Conv2D Layer 3:** 64 filters (3x3 kernel), ReLU activation.
3. **Classification Layers (Dense Head):**
   * **Flatten Layer:** Reshapes the 3D multi-channel feature maps into a 1D vector.
   * **Dense Layer (Hidden):** 64 fully-connected neurons using ReLU activation to learn non-linear combinations of features.
   * **Dense Layer (Output):** 10 output units combined with a `softmax` activation function to generate probability scores across the 10 categorical targets.

## ⚙️ Compilation Setup
* **Optimizer:** Adam (Adaptive Moment Estimation)
* **Loss Function:** `sparse_categorical_crossentropy` (ideal for integer-encoded class labels)
* **Evaluation Metric:** Accuracy

---

## 💻 Tech Stack & Frameworks
* **Language:** Python 3.x
* **Deep Learning Framework:** TensorFlow / Keras
* **Visualization:** Matplotlib
* **Environment:** Jupyter Notebook

## 🚀 Getting Started
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/cifar10-image-classification-cnn.git](https://github.com/your-username/cifar10-image-classification-cnn.git)
   cd cifar10-image-classification-cnn
