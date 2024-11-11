
# **A Basic Vision Transformer (Basic ViT) on MNIST**

This project implements a **Simple Vision Transformer (Simple ViT)** model to classify handwritten digits (0-9) from the MNIST dataset. The model combines patch-based embeddings, self-attention, and an MLP head for efficient and interpretable learning.

---

## **Project Overview**

This repository contains:
- A **patch embedding layer** implemented using `Conv2d` for efficient feature extraction.
- A simplified **self-attention mechanism** to capture dependencies between patches.
- A fully connected **MLP head** for final classification.

The model demonstrates the core principles of the Vision Transformer (ViT) architecture while maintaining simplicity.

---

## **Model Architecture**

### **1. Patch Embedding**
- Input image: (28X28) (MNIST)
- Divided into non-overlapping patches using `Conv2d`:
  - **Patch Size**: Configurable (default: (14X14)).
  - **Input Channels**: 3 (for RGB images, though MNIST is grayscale; adapted for compatibility).
  - **Output Channels**: 64 (embedding dimension).

### **2. Self-Attention**
- **MultiHeadAttention** layer:
  - Captures relationships between patches.
  - Key component for learning inter-patch dependencies.

### **3. MLP Classifier**
- A series of linear layers with **ReLU** activations.
- Final output layer for classification.

---

## **Hyperparameters**

The model allows for flexible hyperparameter tuning, including:
- **Patch Size**: Define the size of each image patch.
- **Learning Rate**: Control the step size for gradient updates.
- **Batch Size**: Define the number of samples per training batch.
- **Epochs**: Set the number of training iterations.
- **Embedding Dimension**: Modify the size of patch embeddings.
- **Number of Attention Heads**: Adjust the number of self-attention heads.
- **Optimizer**: Choose from SGD, Adam, etc.

You can customize these parameters to optimize performance for your specific task.

---

## **Results**

- **Training Accuracy**: ~98%
- **Test Accuracy**: ~96% (Baseline without additional regularization,layer norm).

---

## **Future Improvements**

To improve test performance:
1. **Add Positional Embeddings**:
   - Improve patch location awareness.
2. **Regularization**:
   - Use dropout layers and weight decay to prevent overfitting.
3. **Data Augmentation**:
   - Apply random transformations to improve generalization.

---

## **Usage**

Clone this repository and adjust the hyperparameters, including patch size, as needed to train and evaluate the model.

---
