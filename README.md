# Statlog-Shuttle-Classification-MLP
Classification project using the Statlog (Shuttle) dataset.  Includes preprocessing, feature selection (ANOVA F-test), resampling (manual balancing), Z-score normalization, visualization, and an MLPClassifier for final prediction.

**Authors:** Alejandro Martín Montenegro y Jaime Sánchez Fernández

**Year:** 2025

**Course:** Neural Networks and Deep Learning

---

# Project Overview
This project addresses the classification of the Shuttle Control System into 7 different classes, using a neural network model.
The workflow includes:

- Data exploration and preprocessing
- Balancing strategies for highly imbalanced classes
- Feature selection
- MLP model training
- Cross-validation
- Performance evaluation and interpretation

---

# Dataset Description

The Statlog (Shuttle) dataset consists of: 

- 9 numerical input variables
- A single target variable with 7 output classes
- A total of 58,000+ records
- Strong class imbalance (Class 1 represents ~80% of the dataset)
  
The dataset is available from the UCI Machine Learning Repository.

---
# Libraries Used

- NumPy → numerical computations and array operations
- Pandas → loading, cleaning, merging, and transforming the dataset
- Matplotlib → visualizations (histograms, boxplots, line plots)
- Seaborn → improved statistical visualizations
- Scikit-learn
    - StandardScaler → Z-score normalization
    - LabelEncoder → encoding output classes
    - SelectKBest (ANOVA F-test) → feature selection
    - MLPClassifier → neural network model
    - StratifiedKFold → correct cross-validation
    - train_test_split → data splitting (when required)
    - classification_report, confusion_matrix → evaluation metrics
 
---
# Project Pipeline

1. Data Loading

The dataset comes split into two files: one for training and one for testing.
Before any processing can begin, both parts must be loaded and structured consistently. Pandas ensures proper parsing of large numerical datasets and provides powerful tools for merging and inspecting data.

- Load shuttle.trn and shuttle.tst using Pandas
- Assign column names
- Combine or inspect datasets as needed
- Separate features (X) and labels (y)

2. Exploratory Data Analysis

This part is essential to understand potential data quality issues.
The Shuttle dataset is extremely imbalanced, with class 1 dominating the dataset.
Identifying these characteristics early allows us to design the correct preprocessing steps.
Visual exploration also reveals whether variables have meaningful variance and whether scaling might be required.

- Study class distribution (high imbalance detected)

- Detect duplicate rows

- Ensure correct data types

- Inspect variable ranges and statistics

- Plot histograms and boxplots for all features
