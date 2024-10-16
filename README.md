# Multi-Modal-Fusion
Early Fusion, Late Fusion, and Hybrid Fusion on Emotion Recognition Physiological Data
# Multimodal Fusion Techniques for Machine Learning

This repository contains various multimodal fusion methods for combining data from different modalities (Eye Tracking, GSR, ECG). The repository explores **Early Stage**, **Late Stage**, and **Hybrid Fusion** approaches to improve classification performance on fused datasets.

## Overview
Multimodal fusion aims to combine data from multiple sources to make more accurate predictions by utilizing the strengths of each modality. This project implements fusion techniques using **concatenation**, **autoencoders**, **CNN**, **LSTM**, **PCA**, **SVM**, and **XGBoost** models. It also explores **voting mechanisms** and **attention-weighted approaches** in hybrid models.
![Multimodal Fusion ](https://github.com/user-attachments/assets/871ffe90-31dc-4989-9384-b9cdeb1f4b1c)

### Early Stage Fusion
In early stage fusion, the features from each modality are concatenated at the feature level before classification. Here are the implementations:

- **Auto Encoder Early Stage Fusion**: Combines features from all modalities using autoencoders to reduce dimensionality before classification.
- **CNN Early Stage Fusion**: Applies Convolutional Neural Networks (CNN) for feature extraction and then concatenates these features for classification.
- **Concatenation Early Stage Iterative Imputation Fusion**: Concatenates the features from each modality with iterative imputation for missing values.
- **Concatenation Early Stage KNN Imputation Fusion**: KNN imputation is used for handling missing data before concatenation.
- **Concatenation Early Stage with NaNs Fusion**: Directly concatenates the features from all modalities without imputing NaNs.
- **GCCA Early Stage KNN Imputation Fusion**: Applies Generalized Canonical Correlation Analysis (GCCA) with KNN for imputation before concatenation.
- **LSTM Early Stage Fusion**: Combines features using LSTM networks for sequential data and then concatenates for classification.

### Late Stage Fusion
In late stage fusion, predictions from different models (each trained on one modality) are combined to form the final classification:

- **Auto Encoder Late Stage Fusion**: Autoencoders are applied to each modality, and the final decision is made by combining predictions.
- **CNN Late Stage Fusion**: CNN is applied to each modality separately, and the final classification is achieved by combining predictions.
- **Majority Voting Late Stage Fusion**: Combines the predictions of classifiers for each modality through majority voting.
- **Weighted Averaging Late Stage Fusion**: Combines classifiers' predictions through weighted averaging for final classification.

### Hybrid Fusion
Hybrid fusion uses a combination of early and late stage techniques, often with specialized mechanisms like attention or PCA:

- **Attention-Weighted XGBoost Hybrid Fusion**: Attention mechanism is applied to each modality, and XGBoost is used to classify the attention-weighted features.
- **CNN and LSTM Hybrid Fusion**: Combines CNNs for feature extraction and LSTMs for sequence modeling, followed by classification.
- **PCA and LDA Hybrid Fusion**: Principal Component Analysis (PCA) is used for dimensionality reduction, followed by Linear Discriminant Analysis (LDA) for classification.
- **PCA and SVM Hybrid Fusion**: PCA is used for dimensionality reduction, followed by a Support Vector Machine (SVM) classifier for final classification.

## Data
The dataset includes features extracted from three modalities:
- **Eye Tracking** (`EyeTracking.csv`)
- **GSR** (`GSR.csv`)
- **ECG** (`ECG.csv`)
- Download the dataset from:
- https://www.kaggle.com/datasets/lumaatabbaa/vr-eyes-emotions-dataset-vreed

These files contain time-series and physiological data, which are preprocessed and combined using the above-mentioned fusion techniques.


