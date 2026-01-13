# Network-Anomaly-Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

## Abstract

This project presents a comparative study of Machine Learning and Deep Learning approaches for **Network Intrusion Detection Systems (NIDS)**, specifically focusing on the detection of **Zero-day attacks** in a semi-supervised setting.

Traditional signature-based IDS fail to detect unknown threats. To address this, we implement and evaluate a **Deep Autoencoder** architecture trained exclusively on normal network traffic from the **NSL-KDD** dataset. The system utilizes a **Dynamic Thresholding** mechanism based on reconstruction error distribution to identify anomalies. Performance is benchmarked against established baselines: **Isolation Forest**, **One-Class SVM**, and a **Hybrid Ensemble** model.

## Key Features

* **Semi-supervised Learning:** The model is trained strictly on benign traffic, simulating a realistic zero-day detection scenario.
* **Robust Preprocessing:** Implementation of `log1p` transformation and One-Hot Encoding to handle high-dimensional, skewed network data.
* **Deep Autoencoder Architecture:** Symmetric bottleneck design with **Linear Activation** output to amplify anomaly reconstruction errors.
* **Dynamic Thresholding:** Automated decision boundary determination based on statistical percentiles (e.g., 90th, 95th) of the training loss.
* **Rigorous Benchmarking:** Detailed comparison with Isolation Forest, OCSVM, and Hybrid Ensemble methods.

## Methodology

### 1. System Architecture
The proposed pipeline consists of four main stages: Data Preprocessing, Model Training (Normal Data), Threshold Determination, and Anomaly Detection.

![System Architecture](images/system_architecture.png)
*Figure 1: Overview of the proposed anomaly detection framework.*

### 2. Models Implemented
* **Baseline 1: Isolation Forest:** An ensemble method that isolates anomalies using random partitioning trees. Efficient for high-dimensional data but often yields lower recall.
* **Baseline 2: One-Class SVM (OCSVM):** Maps data to a high-dimensional feature space using an RBF kernel to find a hyperplane enclosing normal data. Offers high accuracy but suffers from cubic computational complexity $O(n^3)$.
* **Proposed: Deep Autoencoder:** A neural network that learns to compress and reconstruct normal traffic. High reconstruction error indicates an anomaly.
* **Hybrid Ensemble:** A parallel voting mechanism combining Isolation Forest and Autoencoder scores.

## Experimental Results

Experiments were conducted on the NSL-KDD dataset. The training set consisted of 100% normal traffic, while the test set included both normal and various attack types (DoS, Probe, R2L, U2R).

### Performance Leaderboard

| Model | ROC-AUC | Precision | Recall | F1-Score | Inference Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **One-Class SVM** | **0.957** | 0.91 | **0.91** | **0.91** | Slow |
| **Deep Autoencoder** | **0.955** | 0.92 | 0.88 | 0.90 | **Fast** |
| Isolation Forest | 0.946 | 0.93 | 0.74 | 0.83 | Fast |
| Hybrid Ensemble | 0.946 | **0.94** | 0.69 | 0.80 | Medium |

### Visualization
The ROC curves below demonstrate the separation capability of each model. While OCSVM marginally outperforms in AUC, the Deep Autoencoder offers the best balance of accuracy and computational efficiency for real-time deployment.

![ROC Comparison](images/roc_comparison.png)
*Figure 2: ROC-AUC Comparison of all implemented models.*

## Project Structure

```text
├── data/                   # NSL-KDD dataset files
├── images/                 # Figures for report and README
├── notebooks/              # Jupyter Notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03a_baseline_models.ipynb
│   └── 03b_deep_autoencoder.ipynb
├── src/                    # Source code for data loader and models
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
