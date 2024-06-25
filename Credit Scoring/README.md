# Loan Default Prediction Project

## Overview
This project aims to predict loan defaults using machine learning techniques. It involves extensive data preprocessing, exploratory data analysis, feature engineering, and model building using a Random Forest classifier.

## Project Structure

### 1. Data Analysis (`data_analysis.py`)
- Data loading and initial preprocessing
- Exploratory Data Analysis (EDA)
- Feature selection and importance analysis

### 2. Model Building (`model.py`)
- Custom data preprocessing pipeline
- Model training using Random Forest
- Model evaluation and cross-validation

## Features
- Handles missing data and outliers
- Addresses skewness in numerical features
- Implements custom transformers for data preprocessing
- Uses ADASYN for handling imbalanced datasets
- Performs cross-validation for robust model evaluation

## Data Preprocessing Steps
1. Drop features with high null percentage
2. Remove irrelevant features
3. Handle missing values using imputation
4. Remove outliers
5. Handle skewness in numerical features
6. Encode categorical variables
7. Scale numerical features

## Model
- Algorithm: Random Forest Classifier
- Hyperparameters:
  - n_estimators: 100
  - max_depth: 10

## Results
- Cross-validation scores: [0.98292285, 0.98084837, 0.98171771, 0.98022894, 0.98089307]
- Mean CV score: 0.9813221906820144
- Training Accuracy: 0.9974496424029429
- Test Accuracy: [To be filled after running on test data]

## Classification Report
[To be filled after running on test data]

## Usage
1. Run `data_analysis.py` for exploratory data analysis and feature selection.
2. Execute `model.py` to preprocess data, train the model, and evaluate its performance.

## Requirements
- Python 3.x
- Libraries: numpy, pandas, scikit-learn, matplotlib, seaborn, imblearn

## Future Improvements
- Experiment with other algorithms (e.g., Gradient Boosting, Neural Networks)
- Fine-tune hyperparameters using techniques like Grid Search or Bayesian Optimization
- Incorporate more advanced feature engineering techniques
- Explore deep learning approaches for potentially better performance

## Contributors
Pranjal Raghava



