# Machine Learning Algorithms Implementation

This repository contains the implementation of various machine learning algorithms and techniques as part of the CSE 6363 Machine Learning course assignments. The primary focus is on Linear Regression and Classification methods, including Logistic Regression and Linear Discriminant Analysis (LDA).

## Table of Contents
- [Project Overview](#project-overview)
- [Implemented Algorithms](#implemented-algorithms)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project involves implementing fundamental machine learning algorithms from scratch and applying them to the Iris dataset. The algorithms include:
- Linear Regression
- Logistic Regression
- Linear Discriminant Analysis (LDA)

## Implemented Algorithms

### Linear Regression
Implemented using a custom `LinearRegression` class with the following methods:
- `fit()`: Trains the model using gradient descent optimization with early stopping.
- `predict()`: Makes predictions based on the trained model.
- `score()`: Calculates the mean squared error to evaluate model performance.
- `save()`: Saves model parameters to a file.
- `load()`: Loads model parameters from a file.

### Classification
1. **Logistic Regression**
   - A supervised learning algorithm for binary classification tasks.
   - Predicts the probability of an instance belonging to a particular class.

2. **Linear Discriminant Analysis (LDA)**
   - A dimensionality reduction technique and classification algorithm.
   - Used for multi-class classification problems.
   - Projects input data onto a lower-dimensional space to maximize class separation.

## Installation
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
