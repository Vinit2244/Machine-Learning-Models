# Statistical Methods in Artificial Intelligence (Monsoon '24)  

**Name:** Vinit Mehta  
**Roll Number:** 2022111001  

---

## About this Repository  

This repository is a culmination of work developed as part of the *Statistical Methods in Artificial Intelligence* (SMAI) course during my 5th semester of the B.Tech. CSE program at IIIT Hyderabad. The mini-project comprises five assignments, each dedicated to exploring and applying diverse machine learning techniques.  

Key aspects of the repository include:  
- **Object-Oriented Design:** All models and methods are implemented using object-oriented programming principles for better modularity and reusability.  
- **Data Preprocessing and Regularization:** Techniques such as feature scaling, handling missing values, and L1/L2 regularization are used to minimize overfitting and enhance model performance.  
- **Hyperparameter Tuning:** Extensive hyperparameter optimization was performed for all models to achieve the best fit. We utilized the Weights and Biases (WandB) library to conduct parameter sweeps and visualize loss graphs.  
- **Visualization and Reporting:** Detailed 2D and 3D visualizations were created for models such as PCA, K-Means, and GMM to better understand their behavior.  
- **Training from Scratch:** Most models were implemented and trained from scratch using libraries like NumPy and Pandas, executed on local machines or Google Colab for computationally intensive tasks.  

> **Note:** Some datasets required GPU acceleration and were processed on Google Colab. For such cases, the data is stored on Google Drive and not in this repository.  

---

## Directory Structure  

- **Assignment Folder:** Contains handler code for each sub-task, along with detailed reports (`README.md`) explaining the methodology and results for each assignment.  
- **Data Folder:** Includes datasets used for training and testing models. For larger datasets processed on Google Colab, the files are stored on Google Drive.  
- **Models Folder:** Contains object-oriented implementations of machine learning models, ensuring scalability and easy debugging.  
- **Performance Metrics Folder:** Houses evaluation functions and utilities for assessing model performance across different metrics.  

---

## Implemented Models  

The repository includes a wide range of machine learning models, categorized as follows:  

### **Classical Machine Learning Models**  
- **K-Nearest Neighbors (KNN):** Optimized with vectorized implementation for faster execution.  
- **Linear Regression:** Includes simple and polynomial regression with 2D and 3D visualizations.  
- **Gaussian Mixture Models (GMM):** Explored AIC and BIC criteria for model selection, with 2D and 3D visualizations.  
- **K-Means Clustering:** Visualized using Elbow plots and WCSS metric; applied in 2D and 3D space.  
- **Principal Component Analysis (PCA):** Used for dimensionality reduction and data analysis, with detailed visualizations.  
- **Hierarchical Clustering:** Tested various distance measures and linkage criteria to achieve optimal clustering.  

### **Deep Learning Models**  
- **Multi-Layer Perceptron (MLP):**  
  - **Classifier:** Implemented for both single-class and multi-class classification.  
  - **Regressor:** Trained on various datasets for regression tasks.  
  - **Hyperparameter Tuning:** Tested multiple loss functions, activation functions, and learning rates using WandB for sweeps and visualizations.  
- **Convolutional Neural Networks (CNN):**  
  - **Classifier:** Designed for image classification tasks like MNIST and Fashion-MNIST.  
  - **Autoencoder:** Built for dimensionality reduction and feature extraction.  
- **Recurrent Neural Networks (RNN):** Tested for generalization, particularly for sequence prediction tasks.  
- **Autoencoders:** Implemented using MLP, CNN, and PCA for effective dimensionality reduction.  
- **Hidden Markov Models (HMM):** Applied for speech recognition tasks using features like Mel-Frequency Cepstral Coefficients (MFCC) and spectrogram analysis.  
- **Kernel Density Estimation (KDE):** Used for estimating data distributions.  

---

## Applications  

The implemented models were applied to solve real-world problems across diverse domains:  

1. **Classification Tasks:**  
   - Music Genre Prediction using KNN, MLP, and K-Means.  
   - Advertisement Classification into multiple categories using MLP.  
   - Fashion Item Classification (Fashion MNIST) using CNN.  
   - Digit Recognition (MNIST) using CNN.  
   - Speech Character Prediction (Spoken Digits Dataset) using HMM.  

2. **Regression Tasks:**  
   - Housing Price Prediction using KNN, MLP, and K-Means.  

3. **Clustering Tasks:**  
   - Word Clustering using KNN, MLP, and K-Means.  

4. **Dimensionality Reduction:**  
   - Autoencoding and feature extraction using MLP, CNN, and PCA.  

5. **Specialized Tasks:**  
   - Wine Quality Prediction using MLP.  
   - Optical Character Recognition using RNN.  
   - Counting the number of bits in binary sequences using RNN.  

---

This repository demonstrates a comprehensive exploration of machine learning techniques, emphasizing practical implementation, performance optimization, and application-driven solutions.