import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib
import time
start_time = time.time()

warnings.filterwarnings("ignore")

print("=" * 60)
print("STROKE PREDICTION - RANDOM FOREST WITH SMOTE")
print("=" * 60)

# Load the Stroke Prediction dataset
print("\n1. LOADING DATASET")
print("-" * 30)
data = pd.read_csv("data/healthcare-dataset-stroke-data.csv")
print(f"Dataset shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print(f"Missing values:\n{data.isnull().sum()}")

# Data preprocessing
print("\n2. DATA PREPROCESSING")
print("-" * 30)
print("Original data shape:", data.shape)
print("Original class distribution:")
print(data['stroke'].value_counts())

# Handle missing BMI values
print("\nHandling missing BMI values...")
bmi = data.groupby(['gender', 'work_type'])['bmi'].transform(lambda x: x.fillna(x.median()))
data['bmi'] = bmi
print(f"Missing values after BMI imputation: {data['bmi'].isnull().sum()}")

# Remove 'Other' gender
print("\nRemoving 'Other' gender category...")
print(f"Gender distribution before: {data['gender'].value_counts()}")
data.drop(data[data['gender'] == 'Other'].index, inplace=True)
print(f"Gender distribution after: {data['gender'].value_counts()}")

# Remove ID column
print("\nRemoving ID column...")
data.drop('id', axis=1, inplace=True)
print(f"Data shape after removing ID: {data.shape}")

# One-Hot Encoding
print("\n3. ONE-HOT ENCODING")
print("-" * 30)
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
print(f"Categorical columns to encode: {categorical_columns}")
print(f"Unique values in each categorical column:")
for col in categorical_columns:
    print(f"  {col}: {data[col].unique()}")

ohe = OneHotEncoder(drop='first')
ohe_data = ohe.fit_transform(data[categorical_columns]).toarray()
ohe_df = pd.DataFrame(ohe_data, columns=ohe.get_feature_names_out(categorical_columns))
data = pd.concat([data.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)
data.drop(columns=categorical_columns, inplace=True)
print(f"Data shape after one-hot encoding: {data.shape}")
print(f"Feature columns: {list(data.columns)}")

# Separate features and target variable
print("\n4. FEATURE AND TARGET SEPARATION")
print("-" * 30)
X = data.drop('stroke', axis=1)
y = data['stroke']
print(f"Features shape (X): {X.shape}")
print(f"Target shape (y): {y.shape}")
print(f"Original class distribution: {Counter(y)}")

# Apply SMOTE
print("\n5. APPLYING SMOTE")
print("-" * 30)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print(f"Resampled features shape (X): {X_resampled.shape}")
print(f"Resampled target shape (y): {y_resampled.shape}")
print(f"Resampled class distribution: {Counter(y_resampled)}")

# Split the data
print("\n6. DATA SPLITTING")
print("-" * 30)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=123)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Training class distribution: {Counter(y_train)}")
print(f"Test class distribution: {Counter(y_test)}")

# Train the Random Forest model
print("\n7. TRAINING RANDOM FOREST MODEL")
print("-" * 30)
model = RandomForestClassifier(n_estimators=600, random_state=42)
print("Training Random Forest with 600 estimators...")
model.fit(X_train, y_train)
print("Training completed!")

# Predictions and evaluation
print("\n8. MODEL EVALUATION")
print("-" * 30)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-Score: {f1 * 100:.2f}%")
print("\nClassification Report:")
print(class_report)
print("\nConfusion Matrix:")
print(conf_matrix)

# Save the model
print("\n9. SAVING MODEL")
print("-" * 30)
model_filename = f"random_forest_model_{accuracy * 100:.2f}%.pkl"
joblib.dump(model, model_filename)
print(f"Model saved as: {model_filename}")

end_time = time.time()
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
print("\n" + "=" * 60)