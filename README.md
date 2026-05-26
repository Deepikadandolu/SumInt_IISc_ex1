# Medical Image Classification using Classical ML and Deep Learning

## Overview

This repository contains the implementation and comparative analysis of multiple machine learning and deep learning techniques for medical image classification using:

- Brain Tumor MRI images
- Chest X-Ray Pneumonia images

The work was carried out through three progressive exercises focused on:

- Support Vector Machines (SVM)
- PCA-based dimensionality reduction
- Convolutional Neural Networks (CNN)
- Transfer Learning
- XGBoost
- VGG16 feature extraction
- Learning-curve analysis under varying training-data sizes

The primary goal of this project is to study how different feature representations and classifiers behave on medical imaging datasets while keeping the testing dataset fixed across experiments.

---


# Datasets Used

## 1. Brain Tumor MRI Dataset

Used for:
- SVM classification
- PCA + SVM
- Raw XGBoost
- PCA + XGBoost
- VGG16 + XGBoost
- CNN transfer learning

Dataset Classes:

- No Tumor
- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor

Dataset Source:

- Brain Tumor MRI Classification Dataset

---

## 2. Chest X-Ray Pneumonia Dataset

Used for:
- SVM classification
- CNN classification
- Raw XGBoost
- PCA + XGBoost
- VGG16 + XGBoost

Dataset Classes:

- NORMAL
- PNEUMONIA

Dataset Source:

- Chest X-Ray Pneumonia Dataset

---

# Repository Contents

| Notebook | Description |
|---|---|
| `brain_tumor_svm.ipynb` | Original PCA + SVM implementation for brain tumor MRI classification |
| `tumor_svm(linear).ipynb` | Linear-kernel SVM implementation |
| `tumor_svm(linear)_split.ipynb` | Linear SVM with varying training percentages |
| `tumor_svm(rbf)_splitdata.ipynb` | RBF-kernel SVM with learning-curve analysis |
| `chest_svm(linear)_split.ipynb` | Chest X-ray SVM classification experiments |
| `Chest_CNN.ipynb` | CNN-based chest X-ray classification |
| `Tumour_transfer_learning.ipynb` | Transfer learning implementation for brain tumor classification |
| `tumour_xgboost.ipynb` | Raw XGBoost for brain tumor dataset |
| `chest_xgboost.ipynb` | Raw XGBoost for chest X-ray dataset |
| `tumour_xgboost_pca.ipynb` | PCA + XGBoost for brain tumor classification |
| `chest_xgboost_vgg16.ipynb` | PCA + XGBoost for chest X-ray classification |
| `tumour_xgboost_vgg16.ipynb` | VGG16 + XGBoost for brain tumor classification |
| `chest_xgboost_vgg16.ipynb` | VGG16 + XGBoost for chest X-ray classification |

---

# Exercise 1 — Brain Tumor Classification using SVM

## Objectives

- Implement SVM-based brain tumor classification
- Compare RBF kernel and Linear kernel
- Study the effect of training data size
- Evaluate PCA dimensionality reduction

## Implemented Pipelines

### PCA + SVM (RBF Kernel)

Pipeline:

```text
MRI Image
    ↓
Grayscale Conversion
    ↓
Flattening
    ↓
PCA Dimensionality Reduction
    ↓
SVM (RBF Kernel)
```

### PCA + SVM (Linear Kernel)

Pipeline:

```text
MRI Image
    ↓
PCA
    ↓
Linear SVM
```

---

## Experimental Study

Training data percentages used:

- 20%
- 40%
- 60%
- 80%

The official test dataset was kept fixed across all experiments to ensure fair evaluation.

---

## Metrics Used

The following metrics were computed:

- Accuracy
- F1 Score
- Sensitivity
- Specificity
- Confusion Matrix

---

## Key Observations

### RBF Kernel

- Captured nonlinear decision boundaries effectively
- Produced higher classification accuracy
- Better suited for complex MRI patterns
- Higher computational complexity

### Linear Kernel

- Faster training
- Lower complexity
- Performed worse on nonlinear tumor distributions
- More interpretable decision boundaries

### Learning Curve Trend

- Testing accuracy improved steadily as training data increased
- Small training sets caused overfitting
- Larger datasets improved generalization capability

---

# Exercise 2 — Chest X-Ray Classification using CNN and SVM

## Objectives

- Repeat Exercise-1 for chest X-ray dataset
- Implement CNN classification
- Apply transfer learning for brain tumor classification
- Compare SVM and CNN performance
- Study training-data scaling behavior

---

# CNN Architecture

The CNN pipeline included:

```text
Input Image
    ↓
Convolution Layers
    ↓
ReLU Activation
    ↓
Max Pooling
    ↓
Flatten
    ↓
Dense Layers
    ↓
Softmax Output
```

---

# Transfer Learning

Transfer learning was implemented using pretrained CNN architectures.

Workflow:

```text
Medical Image
    ↓
Pretrained CNN Feature Extractor
    ↓
Dense Classification Layers
    ↓
Prediction
```

Advantages:

- Faster convergence
- Better feature extraction
- Improved performance on limited medical datasets
- Reduced training time

---

# CNN Experimental Analysis

Training data percentages:

- 20%
- 40%
- 60%
- 80%

The same fixed test dataset was used throughout all CNN and SVM experiments.

---

# CNN Results Trend

Observations:

- CNN performance improved significantly with larger datasets
- Deep learning models benefited more from increased data
- Validation loss decreased with better training coverage
- CNN generalized better than classical ML methods
- Transfer learning improved performance substantially on MRI classification

---

# Exercise 3 — XGBoost and Deep Feature Learning

## Objectives

- Implement Raw XGBoost
- Implement PCA + XGBoost
- Implement VGG16 + XGBoost
- Compare against CNN and SVM methods
- Analyze learning behavior under varying training sizes

---

# Implemented Pipelines

## 1. Raw XGBoost

Pipeline:

```text
Medical Image
    ↓
Flatten Raw Pixels
    ↓
XGBoost Classifier
```

Characteristics:

- High-dimensional input
- Sensitive to noise
- Strong baseline method

---

## 2. PCA + XGBoost

Pipeline:

```text
Medical Image
    ↓
Flattening
    ↓
PCA Dimensionality Reduction
    ↓
XGBoost
```

Characteristics:

- Reduced feature dimensionality
- Lower computational complexity
- Reduced overfitting
- Preserved major variance directions

---

## 3. VGG16 + XGBoost

Pipeline:

```text
Medical Image
    ↓
VGG16 Feature Extraction
    ↓
Global Average Pooling
    ↓
XGBoost Classifier
```

Characteristics:

- Deep semantic feature extraction
- Transfer learning based representation
- Better texture understanding
- Strongest feature representation among all methods

---

# VGG16 Feature Extraction

The VGG16 model was used as a frozen pretrained feature extractor:

```python
base_vgg = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)
```

Global Average Pooling was used to generate compact feature embeddings before XGBoost classification.

---

# Experimental Methodology

For ALL XGBoost experiments:

```text
Official Training Dataset
        ↓
Use 20%, 40%, 60%, 80%
        ↓
Train Model

Official Testing Dataset
        ↓
Fixed Evaluation
```

This ensured:

- No train-test leakage
- Fair comparison across models
- Scientifically valid learning-curve analysis

---

# Experimental Metrics

The following metrics were used consistently across all experiments:

| Metric | Purpose |
|---|---|
| Accuracy | Overall prediction correctness |
| F1 Score | Balance between precision and recall |
| Sensitivity | True positive rate |
| Specificity | True negative rate |
| Confusion Matrix | Error distribution analysis |

---

# Comparative Observations

## Raw XGBoost

- Learned basic intensity patterns
- Worked reasonably on chest X-ray textures
- Struggled with complex tumor morphology
- Higher overfitting tendency

---

## PCA + XGBoost

- Improved stability over raw pixels
- Reduced feature redundancy
- Lower computational complexity
- Some discriminative information lost during PCA compression

---

## VGG16 + XGBoost

- Best overall performance among classical ML pipelines
- Deep features captured semantic image structures
- Strong improvement with increasing training data
- Better generalization capability

---

## CNN

- Strongest end-to-end learning capability
- Improved significantly with more training data
- Better spatial feature understanding
- Higher computational requirements

---

# Learning Curve Analysis

Across all experiments:

- Testing accuracy increased as training data increased
- Small datasets caused higher overfitting
- Larger training datasets improved generalization
- Deep-learning-based approaches scaled better with additional data
- VGG16 + XGBoost consistently outperformed raw feature methods

---

# Key Technical Concepts Explored

This project covers:

- Classical Machine Learning
- Support Vector Machines
- Kernel Methods
- PCA Dimensionality Reduction
- Deep Learning
- CNN Architectures
- Transfer Learning
- XGBoost Classification
- Feature Engineering
- Learning Curve Analysis
- Medical Image Processing
- Model Evaluation Metrics

---

# Libraries and Frameworks

Main libraries used:

```text
Python
NumPy
Pandas
Matplotlib
OpenCV
Scikit-learn
TensorFlow
Keras
XGBoost
KaggleHub
```

---

# Important Experimental Notes

- Official test datasets were kept fixed across all experiments
- Training percentages were varied independently
- PCA was fitted only on training data to avoid leakage
- VGG16 was used as a frozen pretrained feature extractor
- Medical-image-specific metrics such as sensitivity and specificity were included

---

# Results Summary

Overall experimental trend:

```text
Raw XGBoost
        ↓
PCA + XGBoost
        ↓
SVM
        ↓
CNN
        ↓
VGG16 + XGBoost / Transfer Learning
```

The experiments demonstrated the importance of:

- feature representation quality,
- deep feature extraction,
- transfer learning,
- and sufficient training data.

---

# Future Improvements

Potential future extensions:

- Fine-tuning VGG16 layers
- EfficientNet implementation
- ResNet-based transfer learning
- Cross-validation studies
- Grad-CAM visualization
- Attention mechanisms
- Medical segmentation pipelines
- Hyperparameter optimization
- Ensemble learning

---

# References

## SVM Brain Tumor Repository

https://github.com/Adityathere/Brain-Tumor-Detection-Using-SVM

## CNN Chest X-Ray Notebook

https://www.kaggle.com/code/khanfashee/medical-image-classification-for-beginner

## XGBoost Reference

https://medium.com/@tylereyarnell/melanoma-skin-binary-image-classification-using-xgboost-algorithm-f747c1338511

## VGG16 + XGBoost Reference

https://www.kaggle.com/code/sunilgautam/xgboost-with-vgg16-for-image-classification

---

# Author

Deepika D Reddy

Medical Image Classification Comparative Study

SVM • CNN • PCA • XGBoost • Transfer Learning

