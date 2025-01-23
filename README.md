# Rainfall Prediction Project

This project aims to predict rainfall based on weather attributes such as temperature, humidity, wind speed, and other meteorological variables. The goal is to predict whether it will rain tomorrow, using machine learning techniques.


## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Overview](#dataset-overview)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Conclusion](#conclusion)

## Project Overview
This project is divided into two main stages:
1. **Preprocessing**: Clean and prepare the data for model training.
2. **Model Training and Evaluation**: Train machine learning models and evaluate their performance.

The project uses the following techniques:
- **Logistic Regression** as the baseline model.
- **Random Forest** and **XGBoost** for performance improvement.
- Evaluation using classification metrics like **accuracy, precision, recall, and F1-score.**

## Dataset Overview
The dataset contains the following columns:
- **MinTemp**: Minimum temperature in Celsius.
- **MaxTemp**: Maximum temperature in Celsius.
- **Rainfall**: Amount of rainfall in mm.
- **Evaporation**: Evaporation in mm.
- **Sunshine**: Hours of sunshine.
- **WindGustDir, WindDir9am, WindDir3pm**: Wind direction at different times of the day.
- **WindGustSpeed, WindSpeed9am, WindSpeed3pm**: Wind speed at different times.
- **Humidity9am, Humidity3pm**: Humidity at different times.
- **Pressure9am, Pressure3pm**: Atmospheric pressure at different times.
- **Cloud9am, Cloud3pm**: Cloud coverage at different times.
- **Temp9am, Temp3pm**: Temperature at different times.
- **RainToday**: Whether it rained today (Yes/No).
- **RainTomorrow**: Whether it will rain tomorrow (Yes/No).

The target variable for this project is `RainTomorrow`, and the goal is to predict whether it will rain tomorrow based on weather features.
## Steps
### Data Preprocessing
The preprocessing steps are as follows:
- Missing values were handled for both categorical and numerical columns.
- Categorical columns like 'RainToday' and 'RainTomorrow' were label encoded to convert them into numerical format.
- Categorical variables like 'WindGustDir', 'WindDir9am', 'WindDir3pm' were one-hot encoded.
- The numerical columns were scaled using `StandardScaler` to standardize the data for algorithms sensitive to feature scaling.

### Model Training and Evaluation

#### Logistic Regression
- **Accuracy**: 94.59%
- **Classification Report**:
  - **Precision**: 0.95 (for Class 0), 0.93 (for Class 1)
  - **Recall**: 0.98 (for Class 0), 0.81 (for Class 1)
  - **F1-Score**: 0.97 (for Class 0), 0.87 (for Class 1)
- **Confusion Matrix**:
  - True Negatives: 57
  - False Positives: 1
  - False Negatives: 3
  - True Positives: 13

#### Random Forest
- **Accuracy**: 100%
- **Classification Report**:
  - **Precision**: 1.00 (for Class 0), 1.00 (for Class 1)
  - **Recall**: 1.00 (for Class 0), 1.00 (for Class 1)
  - **F1-Score**: 1.00 (for Class 0), 1.00 (for Class 1)
- **Confusion Matrix**:
  - True Negatives: 58
  - False Positives: 0
  - False Negatives: 0
  - True Positives: 16

#### XGBoost
- **Accuracy**: 100%
- **Classification Report**:
  - **Precision**: 1.00 (for Class 0), 1.00 (for Class 1)
  - **Recall**: 1.00 (for Class 0), 1.00 (for Class 1)
  - **F1-Score**: 1.00 (for Class 0), 1.00 (for Class 1)
- **Confusion Matrix**:
  - True Negatives: 58
  - False Positives: 0
  - False Negatives: 0
  - True Positives: 16

#### Model Performance Comparison:
| Model            | Accuracy  | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) |
|------------------|-----------|---------------|---------------|------------|------------|--------------|--------------|
| Logistic Regression | 94.59%  | 0.95          | 0.93          | 0.98       | 0.81       | 0.97         | 0.87         |
| Random Forest       | 100%    | 1.00          | 1.00          | 1.00       | 1.00       | 1.00         | 1.00         |
| XGBoost             | 100%    | 1.00          | 1.00          | 1.00       | 1.00       | 1.00         | 1.00         |

#### Conclusion:
- **XGBoost** and **Random Forest** performed exceptionally well with **100% accuracy**, while **Logistic Regression** achieved **94.59% accuracy**.
- We observed that **Random Forest** and **XGBoost** produced perfect precision, recall, and F1-scores, indicating no false positives or false negatives for these models.
- Logistic Regression, though effective, showed slight variance in performance between the two classes, particularly for the minority class (1).

