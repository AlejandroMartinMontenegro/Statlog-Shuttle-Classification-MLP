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

**1. Data Loading**

The dataset comes split into two files: one for training and one for testing.
Before any processing can begin, both parts must be loaded and structured consistently. Pandas ensures proper parsing of large numerical datasets and provides powerful tools for merging and inspecting data.

- Load shuttle.trn and shuttle.tst using Pandas
- Assign column names
- Combine or inspect datasets as needed
- Separate features (X) and labels (y)

**2. Exploratory Data Analysis**

This part is essential to understand potential data quality issues.
The Shuttle dataset is extremely imbalanced, with class 1 dominating the dataset.
Identifying these characteristics early allows us to design the correct preprocessing steps.
Visual exploration also reveals whether variables have meaningful variance and whether scaling might be required.

- Study class distribution (high imbalance detected)
- Detect duplicate rows
- Ensure correct data types
- Inspect variable ranges and statistics
- Plot histograms and boxplots for all features

**3. Preprocessing**

- Remove extremely rare classes

- Manually balance classes 1, 4, and 5

- Normalize all variables using Z-score scaling

- Encode labels using LabelEncoder

Removing rare classes helps avoid unstable training caused by classes with negligible representation.
Manual balancing ensures that each remaining class contributes equally during training; otherwise, the neural network would be biased toward the dominant class.
Z-score normalization ensures all variables are centered and scaled, which is critical for neural networks to avoid gradient explosion or vanishing.
Label encoding converts categorical class labels into integers, which MLPClassifier expects.

**4. Train/Test**

- Use Stratified 5-Fold Cross-Validation

- Maintain class proportions across folds

- Evaluate model stability and generalization

With an originally imbalanced dataset, it is extremely important to ensure that each fold of cross-validation reflects the true class distribution.
Using StratifiedKFold guarantees this balance and prevents misleading validation results.
Cross-validation also measures how consistently the model performs across multiple splits.

**Model Training - MLPClassifier**

- Neural network model with hidden layers

- ReLU activation function

- Adam optimizer

- Early stopping enabled

- Train on the balanced dataset

- Monitor convergence and loss curve

The Multilayer Perceptron is a powerful non-linear classifier.
The combination of ReLU activation and the Adam optimizer allows the model to converge quickly and handle complex input spaces.
Training on the manually balanced dataset ensures the model learns each class equally.
Early stopping prevents overfitting by halting training when improvements stall.


**Model Evaluation**
- Generate accuracy, precision, recall, and F1-score

- Build a confusion matrix

- Analyze mistakes class-by-class

- Verify stability across folds

Evaluation verifies not only how well the model performs, but also how the performance is distributed across classes.
Because this is a multiclass problem, metrics such as precision, recall, and F1-score become essential—not only overall accuracy.
The confusion matrix is the most informative tool: if it is nearly diagonal, the model is performing very well.

**Results**


Performance Metrics
| Metric        | Class 1 | Class 4 | Class 5 | Macro Average | Overall    |
| ------------- | ------- | ------- | ------- | ------------- | ---------- |
| **Accuracy**  | –       | –       | –       | –             | **99.99%** |
| **Precision** | 99.99%  | 99.98%  | 99.99%  | **99.99%**    | –          |
| **Recall**    | 100%    | 99.98%  | 99.98%  | **99.99%**    | –          |
| **F1-Score**  | 100%    | 99.98%  | 99.98%  | **99.99%**    | –          |

Confusion Matrix
| Actual \ Predicted | Class 1  | Class 4  | Class 5  |
| ------------------ | -------- | -------- | -------- |
| **Class 1**        | **2458** | 1        | 0        |
| **Class 4**        | 0        | **2457** | 1        |
| **Class 5**        | 0        | 1        | **2457** |

The model achieves near-perfect performance, with accuracy, precision, recall, and F1-score all exceeding 99.9%. The confusion matrix shows minimal misclassification, indicating that the Shuttle dataset becomes highly separable once properly balanced and normalized. The combination of manual resampling, ANOVA feature selection, and MLP training produces a stable and extremely effective classifier.

---
# Repository Structure

Statlog-Shuttle-Classification-MLP/
│
├── ProyectoFinal1.ipynb                     # Main Jupyter Notebook containing the full workflow
├── Proyecto Final.docx                      # Written project report
│
├── shuttle.trn                              # Original training file
├── shuttle.tst                              # Original test file
├── shuttle.trn.Z                            # Compressed version of the training file
│
├── shuttle_train_balanceado_manual.csv       # Manually balanced dataset (classes 1, 4, 5)
├── shuttle_train_preprocesado_balanceado.csv # Final preprocessed + balanced dataset
│
└── README.md                                 # Project documentation

---
# How to run the project

**1. Clone repository**
git clone https://github.com/<your-username>/Statlog-Shuttle-Classification-MLP.git
cd Statlog-Shuttle-Classification-MLP

**2. Install required libraries**
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

**3. Launch Jupyter Notebook**
jupyter notebook




