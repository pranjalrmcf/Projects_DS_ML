# Player Position and Rating Prediction

This project builds a machine learning model to predict a player's position and overall rating based on various attributes from the FIFA dataset.

## Table of Contents

- [Introduction](#introduction)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling](#modeling)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to predict the position and rating of football players using machine learning techniques. The dataset used contains various attributes of players from FIFA, including physical attributes, skills, and performance metrics.

## Data

The dataset used in this project is from the FIFA video game series and includes the following key features:

- Player positions
- Overall rating
- Age
- League level
- Various skill and physical attributes

Data Preprocessing & Modeling
The data preprocessing steps include:

Dropping irrelevant columns
Handling missing values
Splitting composite columns
Encoding categorical variables
Standardizing numerical features
Model Training
Position Prediction
Encode the target variable (position_4).
Train a Random Forest Classifier to predict the position category (Forward, Midfielder, Defender, Goalkeeper).
Evaluate the model using accuracy, precision, recall, and confusion matrix.
Rating Prediction
Train a Random Forest Regressor to predict the overall player rating.
Evaluate the model using Mean Squared Error (MSE) and R-squared (R²).

Results
Position Prediction
Accuracy: 92.08%
Precision: [0.95, 0.89, 1.00, 0.88]
Recall: [0.96, 0.84, 1.00, 0.90]
Rating Prediction
Mean Squared Error (MSE): 0.49
R-squared (R²): 0.99
