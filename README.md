# Customer Churn Prediction using Artificial Neural Networks (ANN)

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Installation](#installation) 
- [Usage](#usage)
- [Future Work](#future-work)

## Project Overview
This project aims to predict customer churn in a telecommunication company using an Artificial Neural Network (ANN) model. Customer churn is a critical issue for businesses, and predicting it can help companies take proactive measures to retain valuable customers.

The notebook covers data loading, extensive data preprocessing, exploratory data analysis, building and training a simple ANN model using TensorFlow/Keras, and evaluating its performance.

## Dataset
The dataset used for this project is `tel_churn.csv`, which contains various customer attributes and their churn status. Key features include:
- `customerID`: Unique identifier for each customer.
- `gender`: Customer's gender.
- `SeniorCitizen`: Indicates if the customer is a senior citizen (1) or not (0).
- `Partner`: Indicates if the customer has a partner (Yes/No).
- `Dependents`: Indicates if the customer has dependents (Yes/No).
- `tenure`: Number of months the customer has stayed with the company.
- `PhoneService`: Indicates if the customer has phone service (Yes/No).
- `MultipleLines`: Indicates if the customer has multiple lines (Yes/No/No phone service).
- `InternetService`: Customer's internet service provider (DSL/Fiber optic/No).
- `OnlineSecurity`: Indicates if the customer has online security (Yes/No/No internet service).
- `OnlineBackup`: Indicates if the customer has online backup (Yes/No/No internet service).
- `DeviceProtection`: Indicates if the customer has device protection (Yes/No/No internet service).
- `TechSupport`: Indicates if the customer has tech support (Yes/No/No internet service).
- `StreamingTV`: Indicates if the customer has streaming TV (Yes/No/No internet service).
- `StreamingMovies`: Indicates if the customer has streaming movies (Yes/No/No internet service).
- `Contract`: The customer's contract type (Month-to-month/One year/Two year).
- `PaperlessBilling`: Indicates if the customer has paperless billing (Yes/No).
- `PaymentMethod`: The customer's payment method.
- `MonthlyCharges`: The amount charged to the customer monthly.
- `TotalCharges`: The total amount charged to the customer.
- `Churn`: Indicates if the customer churned (Yes/No). This is the target variable.

## Data Preprocessing
The following preprocessing steps were performed:
1.  **Drop `customerID`**: This column is not relevant for model training.
2.  **Handle `TotalCharges`**: Converted to numeric type. Rows with empty strings were identified and removed.
3.  **Label Encoding**: Categorical features like `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `PaperlessBilling`, and `Churn` were converted to numerical representations (0 and 1).
4.  **One-Hot Encoding**: Features with more than two unique categorical values (`InternetService`, `Contract`, `PaymentMethod`) were one-hot encoded using `pd.get_dummies()`.
5.  **Scaling**: Numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) were scaled using `MinMaxScaler` to normalize their ranges.

## Exploratory Data Analysis (EDA)
Visualizations were used to understand the distribution of `tenure` and `MonthlyCharges` with respect to churn status. Histograms were generated to show the number of customers churned vs. not churned based on these features.

## Model Architecture
The ANN model was built using TensorFlow/Keras with the following architecture:
- **Input Layer**: `Dense` layer with 64 units, ReLU activation, and `input_shape` equal to the number of features in `X_train`.
- **Hidden Layer**: `Dense` layer with 32 units and ReLU activation.
- **Output Layer**: `Dense` layer with 1 unit and Sigmoid activation for binary classification.

## Training and Evaluation
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 100
- **Validation Data**: 20% of the data was used for validation (`X_test`, `y_test`).

The model was trained on the preprocessed data, and its performance was evaluated using `classification_report` and a confusion matrix.

## Results
The model achieved the following performance on the test set:
- **Accuracy**: Approximately 74%
- A detailed `classification_report` showing precision, recall, and f1-score for both churn classes was generated.
- A confusion matrix visualized with a heatmap provides a clear view of true positives, true negatives, false positives, and false negatives.

## Installation
To run this notebook, you'll need the following libraries. You can install them using pip:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
```

## Usage
1.  Clone this repository:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  Ensure you have the `tel_churn.csv` dataset in the same directory as the notebook.
3.  Open the Jupyter notebook (or Colab) and run all cells.


