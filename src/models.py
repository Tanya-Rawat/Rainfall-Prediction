import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Function to load preprocessed data
def load_data():
    X_train = pd.read_csv('data/processed_X_train.csv')
    X_test = pd.read_csv('data/processed_X_test.csv')
    y_train = pd.read_csv('data/processed_y_train.csv')
    y_test = pd.read_csv('data/processed_y_test.csv')
    return X_train, X_test, y_train, y_test

# Function to train Logistic Regression model
def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, accuracy

# Function to train Random Forest model
def train_random_forest(X_train, y_train, X_test, y_test):
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return rf_model, accuracy

# Function to train XGBoost model
def train_xgboost(X_train, y_train, X_test, y_test):
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return xgb_model, accuracy

# Main function to load data, train models, and compare performance
def main():
    # Load preprocessed data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train and evaluate Logistic Regression
    print("\nTraining Logistic Regression...")
    logistic_model, logistic_accuracy = train_logistic_regression(X_train, y_train, X_test, y_test)
    
    # Train and evaluate Random Forest
    print("\nTraining Random Forest...")
    rf_model, rf_accuracy = train_random_forest(X_train, y_train, X_test, y_test)
    
    # Train and evaluate XGBoost
    print("\nTraining XGBoost...")
    xgb_model, xgb_accuracy = train_xgboost(X_train, y_train, X_test, y_test)
    
    # Compare the models' accuracies
    print("\nModel Performance Comparison:")
    print(f"Logistic Regression Accuracy: {logistic_accuracy:.4f}")
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

if __name__ == "__main__":
    main()
