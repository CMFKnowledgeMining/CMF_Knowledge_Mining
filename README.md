# Crash Modification Factor (CMF) Prediction Model

## Overview
This repository contains the code for a Python-based predictive modeling framework designed to estimate Crash Modification Factors (CMFs). CMFs quantify the effect of roadway or intersection modifications on crash frequencies. The model leverages machine learning algorithms, including Random Forest and Neural Networks, to predict CMFs based on roadway and intersection characteristics.

## Dependencies
- Python 3.x
- PyTorch
- scikit-learn
- pandas
- numpy
- sentence_transformers
- matplotlib
- seaborn
- joblib
- mlinsights
- scipy
## Hardware Requirements
### GPU Recommendations
While it is possible to run the model on a CPU, using a GPU can significantly speed up model training and inference times, especially with large datasets or complex deep learning models. Here are our recommendations for GPU hardware:

- **Minimum**: NVIDIA GTX 1060 or equivalent
- **Recommended**: NVIDIA RTX 3060 or higher for faster processing and training.

### CUDA Compatibility
Ensure that your GPU is compatible with CUDA 11.0 or higher to fully leverage PyTorch functionalities for GPU acceleration.

## Model Training
### Data Preparation
The CMF Clearinghouse Data can be accessible at https://www.cmfclearinghouse.org/cmf_data.php

Split the data into 80-20 ratio for nested cross-validation 

### Model training steps
Step1: Fine-tune the pre-trained Sentence Transformer to get the semantic encoder 

Step2: Train the MLP regression model to predict CMFs 

### Model Evaluation
Assess the model's performance using evaluate_metric_cross_validation

