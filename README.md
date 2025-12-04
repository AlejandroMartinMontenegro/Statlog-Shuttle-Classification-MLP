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

