# 📘 MNIST Digit Classification Using K-Nearest Neighbors (KNN) and PCA  
### *A Comprehensive Machine Learning Workflow with Dimensionality Reduction*

This repository contains a fully documented Jupyter Notebook demonstrating **handwritten digit classification** on the **MNIST dataset** using the **k-Nearest Neighbors (KNN)** algorithm, enhanced with **Principal Component Analysis (PCA)** for dimensionality reduction. The notebook provides a structured, pedagogical pipeline aligned with academic machine learning methodology.

GitHub Repository:  
👉 **https://github.com/sajjadbaghaee/KNN-mnist_classification**

---

## 🔍 1. Overview

The MNIST dataset consists of **70,000 grayscale images**, each representing a handwritten digit (0–9). Each image is 28×28 pixels, flattened into a 784-dimensional vector. KNN is a simple, non-parametric classifier, but its performance is heavily influenced by high dimensionality and computational cost.

This notebook demonstrates:

- Exploratory data analysis and visualization  
- Training a baseline KNN classifier  
- Evaluating model accuracy and misclassifications  
- Understanding the effect of high dimensionality  
- Applying PCA to compress MNIST into a more meaningful feature space  
- Comparing accuracy and performance with and without PCA  
- Visualizing PCA components (“eigen-digits”) and 2D projections  

The goal is to showcase both **KNN’s strengths** and **PCA’s importance** in handling large, high-dimensional datasets.

---

## 📁 2. Repository Contents

```
.
├── mnist_knn_classification.ipynb     # Main notebook: KNN + PCA workflow
├── README.md                          # Project documentation (this file)
└── (optional future folders)
    ├── images/                        # Stored plots (variance curves, PCA digits, etc.)
    └── src/                           # Helper scripts (if extended later)
```

---

## 📊 3. Methods and Workflow

### **3.1 Data Loading & Preprocessing**
- MNIST dataset is loaded from OpenML using scikit-learn.  
- Pixel values are normalized to \[0,1\] for stable distance computation.  
- Subsets (20,000 train / 5,000 test) improve computational efficiency.  

### **3.2 Exploratory Data Analysis (EDA)**
Includes:
- Visualization of sample digits  
- Grid plots of random samples  
- Sanity checks of shapes and distributions  

---

### **3.3 Baseline KNN Classifier**
With `k = 3`, the notebook evaluates:

- Accuracy  
- Confusion matrix  
- Classification report  
- Prediction time  
- Examples of correct and incorrect classifications  

This forms a reference point for later comparison.

---

### **3.4 Hyperparameter Sensitivity**
Performance is evaluated for odd values of `k` from 1 to 11:

- Training accuracy  
- Test accuracy  
- Runtime characteristics  

Revealing KNN’s natural **bias–variance trade-off**.

---

### **3.5 Principal Component Analysis (PCA)**
PCA is used to address:

- Curse of dimensionality  
- Noisy and redundant pixel features  
- Slow KNN computation in 784-D space  

The notebook includes:

- Explained variance curves  
- Automatic PCA component selection (95% variance)  
- Visualization of PCA components (eigen-digits)  
- PCA 2D projection of MNIST clusters  

---

### **3.6 PCA + KNN Pipeline**
The notebook compares multiple models:

| Model          | Accuracy | Fit Time | Prediction Time |
|----------------|----------|----------|------------------|
| No PCA         | High     | Slow     | Very Slow        |
| PCA-20         | Medium   | Fast     | Very Fast        |
| PCA-50         | High     | Fast     | Fast             |
| PCA-100        | High     | Moderate | Moderate         |

Results demonstrate that **50 PCA components** deliver an excellent trade-off between accuracy and speed.

---

## 🧠 4. Key Insights

- PCA significantly improves KNN performance by reducing noise and dimensionality.  
- PCA speeds up prediction dramatically (10–15× faster).  
- Many MNIST pixels are redundant, and meaningful information lies in a lower-dimensional manifold.  
- PCA components visualize key features such as strokes, curves, and digit structure.  

---

## 🧰 5. Requirements

Install dependencies using:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## ▶️ 6. Running the Notebook

1. Clone the repository:
```bash
git clone https://github.com/sajjadbaghaee/KNN-mnist_classification.git
```

2. Navigate to the project:
```bash
cd KNN-mnist_classification
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Open **mnist_knn_classification.ipynb** and run the cells.

---

## 🎯 7. Learning Objectives

This project teaches:

- Fundamental handling of image datasets  
- Distance-based classification (KNN)  
- Impact of dimensionality on ML algorithms  
- Application and interpretation of PCA  
- Practical model evaluation techniques  

---

## 🔮 8. Future Work

Potential extensions:

- Logistic Regression on MNIST  
- SVM with RBF kernels  
- Random Forest / XGBoost baselines  
- Neural network (MLP/CNN) comparison  
- t-SNE and UMAP visualizations  
- Approximate nearest-neighbor search (FAISS, Annoy)

<p align="center">
  <strong> 🔥📘💻 More Codes and Tutorials are available at:</strong><br>
  <a href="https://github.com/sajjadbaghaee">
    <img src="https://img.shields.io/badge/GitHub-sajjadbaghaee-blue?logo=github">
  </a>
</p>
---

## 📄 9. License

This project is released under the **MIT License**, permitting unrestricted academic and educational use.
