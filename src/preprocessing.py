import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
data = pd.read_csv('data/weather.csv')

# Handle missing values in numerical columns
numerical_features = ['Sunshine', 'WindGustSpeed', 'WindSpeed9am']
data[numerical_features] = data[numerical_features].fillna(data[numerical_features].median())

# Handle missing values in categorical columns
data['WindDir9am'] = data['WindDir9am'].fillna(data['WindDir9am'].mode()[0])
data['WindDir3pm'] = data['WindDir3pm'].fillna(data['WindDir3pm'].mode()[0])
data['WindGustDir'] = data['WindGustDir'].fillna(data['WindGustDir'].mode()[0])

# Checking for missing values
print(data.isnull().sum())

category_feature =[feature for feature in data.columns if data[feature].dtypes == 'O']
le = LabelEncoder()
for feature in category_feature:
    data[feature] = le.fit_transform(data[feature])
print(data.head())

# Split data into features and target
X = data.drop(columns=['RainTomorrow'])
y = data['RainTomorrow']

# Split data into features and target
X = data.drop(columns=['RainTomorrow'])
y = data['RainTomorrow']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
scaler.fit(X_train)  # Fit the scaler to the training data

X_train_scaled = scaler.transform(X_train)  # Transform training data
X_test_scaled = scaler.transform(X_test)    # Transform test data

# Save the processed data
pd.DataFrame(X_train_scaled).to_csv('data/processed_X_train.csv', index=False)
pd.DataFrame(X_test_scaled).to_csv('data/processed_X_test.csv', index=False)
y_train.to_csv('data/processed_y_train.csv', index=False)
y_test.to_csv('data/processed_y_test.csv', index=False)

print("Preprocessing completed and data saved.")