import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

class StrokeDataPreprocessor:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.encoder = None
        self.feature_selector = None
        self.model = None
        
    def load_data(self, filepath):
        """Load and display initial data information"""
        print("=" * 80)
        print("🔍 STEP 1: DATA LOADING AND INITIAL EXPLORATION")
        print("=" * 80)
        
        self.data = pd.read_csv(filepath)
        
        # Drop the 'id' column as it's not useful for prediction
        if 'id' in self.data.columns:
            self.data.drop('id', axis=1, inplace=True)
            print("🗑️  Dropped 'id' column as it's not useful for prediction")
        
        print(f"📊 Dataset Shape: {self.data.shape}")
        print(f"📋 Columns: {list(self.data.columns)}")
        print(f"📈 Data Types:\n{self.data.dtypes}")
        
        # Display missing values
        missing_values = self.data.isnull().sum()
        missing_percentage = (missing_values / len(self.data)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing_values,
            'Missing_Percentage': missing_percentage
        })
        print(f"\n❌ Missing Values Analysis:")
        print(missing_df[missing_df['Missing_Count'] > 0])
        
        # Display basic statistics
        print(f"\n📊 Basic Statistics:")
        print(self.data.describe())
        
        return self.data
    
    def explore_target_distribution(self):
        """Analyze target variable distribution"""
        print("\n" + "=" * 80)
        print("🎯 STEP 2: TARGET VARIABLE ANALYSIS")
        print("=" * 80)
        
        target_counts = self.data['stroke'].value_counts()
        target_percentages = (target_counts / len(self.data)) * 100
        
        print(f"🎯 Target Distribution:")
        print(f"   No Stroke (0): {target_counts[0]} samples ({target_percentages[0]:.2f}%)")
        print(f"   Stroke (1): {target_counts[1]} samples ({target_percentages[1]:.2f}%)")
        
        # Calculate imbalance ratio
        imbalance_ratio = target_counts[0] / target_counts[1]
        print(f"📊 Imbalance Ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 10:
            print("⚠️  WARNING: Severe class imbalance detected!")
        elif imbalance_ratio > 5:
            print("⚠️  WARNING: Moderate class imbalance detected!")
        else:
            print("✅ Class distribution is relatively balanced")
    
    def handle_missing_values(self):
        """Handle missing values with detailed analysis"""
        print("\n" + "=" * 80)
        print("🔧 STEP 3: MISSING VALUES HANDLING")
        print("=" * 80)
        
        missing_before = self.data.isnull().sum()
        print(f"📊 Missing values before handling:")
        print(missing_before[missing_before > 0])
        
        # Handle BMI missing values with group-based imputation
        if 'bmi' in self.data.columns and self.data['bmi'].isnull().sum() > 0:
            print(f"\n🔍 BMI missing values: {self.data['bmi'].isnull().sum()}")
            
            # Group by gender and work_type for more accurate imputation
            bmi_median_by_group = self.data.groupby(['gender', 'work_type'])['bmi'].median()
            print(f"📊 BMI median by gender and work type:")
            print(bmi_median_by_group)
            
            # Fill missing BMI values with group median
            self.data['bmi'] = self.data.groupby(['gender', 'work_type'])['bmi'].transform(
                lambda x: x.fillna(x.median())
            )
            
            # If still missing, fill with overall median
            if self.data['bmi'].isnull().sum() > 0:
                overall_median = self.data['bmi'].median()
                self.data['bmi'].fillna(overall_median, inplace=True)
                print(f"🔧 Filled remaining missing BMI with overall median: {overall_median:.2f}")
        
        missing_after = self.data.isnull().sum()
        print(f"\n✅ Missing values after handling:")
        print(missing_after[missing_after > 0])
        
        if missing_after.sum() == 0:
            print("🎉 All missing values have been successfully handled!")
    
    def remove_outliers_and_anomalies(self):
        """Remove outliers and handle anomalies"""
        print("\n" + "=" * 80)
        print("🚫 STEP 4: OUTLIER AND ANOMALY DETECTION")
        print("=" * 80)
        
        # Remove 'Other' gender category (anomaly)
        gender_counts = self.data['gender'].value_counts()
        print(f"👥 Gender distribution before cleaning:")
        print(gender_counts)
        
        if 'Other' in self.data['gender'].values:
            self.data = self.data[self.data['gender'] != 'Other']
            print(f"🗑️  Removed 'Other' gender category")
        
        print(f"👥 Gender distribution after cleaning:")
        print(self.data['gender'].value_counts())
        
        # Detect outliers in numerical columns
        numerical_cols = ['age', 'avg_glucose_level', 'bmi']
        
        for col in numerical_cols:
            if col in self.data.columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
                print(f"\n📊 {col.upper()} Outlier Analysis:")
                print(f"   Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
                print(f"   Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
                print(f"   Outliers found: {len(outliers)} ({len(outliers)/len(self.data)*100:.2f}%)")
                
                if len(outliers) > 0:
                    print(f"   Outlier range: {outliers[col].min():.2f} - {outliers[col].max():.2f}")
        
        print(f"\n📊 Final dataset shape after cleaning: {self.data.shape}")
    
    def feature_engineering(self):
        """Create new features and transformations"""
        print("\n" + "=" * 80)
        print("🔧 STEP 5: FEATURE ENGINEERING")
        print("=" * 80)
        
        # Create age groups
        self.data['age_group'] = pd.cut(self.data['age'], 
                                       bins=[0, 30, 45, 60, 75, 100], 
                                       labels=['Young', 'Adult', 'Middle-aged', 'Senior', 'Elderly'])
        
        # Create BMI categories
        self.data['bmi_category'] = pd.cut(self.data['bmi'],
                                          bins=[0, 18.5, 25, 30, 100],
                                          labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        
        # Create glucose level categories
        self.data['glucose_category'] = pd.cut(self.data['avg_glucose_level'],
                                              bins=[0, 100, 125, 200, 1000],
                                              labels=['Normal', 'Prediabetes', 'Diabetes', 'Very High'])
        
        # Create enhanced risk score with more factors
        risk_factors = 0
        # Age factors (more granular)
        risk_factors += (self.data['age'] > 65).astype(int) * 2  # Elderly gets 2 points
        risk_factors += (self.data['age'] > 75).astype(int) * 1  # Very elderly gets extra point
        
        # Medical conditions
        risk_factors += self.data['hypertension'] * 2  # Hypertension gets 2 points
        risk_factors += self.data['heart_disease'] * 2  # Heart disease gets 2 points
        
        # Glucose levels (more granular)
        risk_factors += (self.data['avg_glucose_level'] > 140).astype(int) * 1  # High glucose
        risk_factors += (self.data['avg_glucose_level'] > 200).astype(int) * 1  # Very high glucose
        
        # BMI factors (more granular)
        risk_factors += (self.data['bmi'] > 30).astype(int) * 1  # Obese
        risk_factors += (self.data['bmi'] > 40).astype(int) * 1  # Severely obese
        
        # Smoking status
        risk_factors += (self.data['smoking_status'] == 'smokes').astype(int) * 1
        risk_factors += (self.data['smoking_status'] == 'formerly smoked').astype(int) * 1
        
        self.data['risk_score'] = risk_factors
        
        print("🔧 New features created:")
        print("   - age_group: Categorical age groups")
        print("   - bmi_category: BMI classification")
        print("   - glucose_category: Glucose level classification")
        print("   - risk_score: Composite risk score (0-5)")
        
        print(f"\n📊 Risk Score Distribution:")
        print(self.data['risk_score'].value_counts().sort_index())
        
        print(f"\n📊 Age Group Distribution:")
        print(self.data['age_group'].value_counts())
        
        print(f"\n📊 BMI Category Distribution:")
        print(self.data['bmi_category'].value_counts())
    
    def encode_categorical_variables(self):
        """Encode categorical variables with detailed analysis"""
        print("\n" + "=" * 80)
        print("🔤 STEP 6: CATEGORICAL VARIABLE ENCODING")
        print("=" * 80)
        
        categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        
        print("📊 Categorical variables analysis:")
        for col in categorical_columns:
            if col in self.data.columns:
                unique_values = self.data[col].unique()
                value_counts = self.data[col].value_counts()
                print(f"\n🔤 {col.upper()}:")
                print(f"   Unique values: {unique_values}")
                print(f"   Value counts:")
                for val, count in value_counts.items():
                    percentage = (count / len(self.data)) * 100
                    print(f"     {val}: {count} ({percentage:.2f}%)")
        
        # One-Hot Encoding
        print(f"\n🔧 Applying One-Hot Encoding to: {categorical_columns}")
        
        # Create encoder
        self.encoder = OneHotEncoder(drop='first', sparse_output=False)
        
        # Fit and transform
        encoded_data = self.encoder.fit_transform(self.data[categorical_columns])
        encoded_df = pd.DataFrame(encoded_data, 
                                columns=self.encoder.get_feature_names_out(categorical_columns))
        
        # Combine with original data
        self.data = pd.concat([self.data.reset_index(drop=True), 
                             encoded_df.reset_index(drop=True)], axis=1)
        
        # Drop original categorical columns
        self.data.drop(columns=categorical_columns, inplace=True)
        
        print(f"✅ Encoding completed!")
        print(f"📊 New feature columns: {list(self.data.columns)}")
        print(f"📊 Final dataset shape: {self.data.shape}")
    
    def feature_selection(self):
        """Perform feature selection"""
        print("\n" + "=" * 80)
        print("🎯 STEP 7: FEATURE SELECTION")
        print("=" * 80)
        
        # Separate features and target
        X = self.data.drop(['stroke', 'age_group', 'bmi_category', 'glucose_category'], axis=1)
        y = self.data['stroke']
        
        # Use SelectKBest for feature selection
        self.feature_selector = SelectKBest(score_func=f_classif, k=15)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        
        print(f"🎯 Selected {len(selected_features)} features:")
        for i, feature in enumerate(selected_features, 1):
            score = self.feature_selector.scores_[self.feature_selector.get_support()][i-1]
            print(f"   {i:2d}. {feature}: {score:.4f}")
        
        # Update data with selected features
        self.data = pd.concat([pd.DataFrame(X_selected, columns=selected_features), y], axis=1)
        
        print(f"✅ Feature selection completed!")
        print(f"📊 Final feature set: {selected_features}")
    
    def apply_smote_balancing(self):
        """Apply SMOTE for class balancing"""
        print("\n" + "=" * 80)
        print("⚖️  STEP 8: CLASS BALANCING WITH SMOTE")
        print("=" * 80)
        
        # Separate features and target
        X = self.data.drop('stroke', axis=1)
        y = self.data['stroke']
        
        print(f"📊 Before SMOTE:")
        print(f"   Dataset shape: {X.shape}")
        print(f"   Class distribution: {Counter(y)}")
        
        # Apply SMOTE
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"\n📊 After SMOTE:")
        print(f"   Dataset shape: {X_resampled.shape}")
        print(f"   Class distribution: {Counter(y_resampled)}")
        
        # Store resampled data
        self.X_resampled = X_resampled
        self.y_resampled = y_resampled
        
        print("✅ SMOTE balancing completed!")
    
    def split_data(self):
        """Split data into train and test sets"""
        print("\n" + "=" * 80)
        print("✂️  STEP 9: DATA SPLITTING")
        print("=" * 80)
        
        # Split the resampled data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_resampled, self.y_resampled, 
            test_size=0.2, random_state=42, stratify=self.y_resampled
        )
        
        print(f"📊 Training set: {self.X_train.shape}")
        print(f"📊 Test set: {self.X_test.shape}")
        print(f"📊 Training class distribution: {Counter(self.y_train)}")
        print(f"📊 Test class distribution: {Counter(self.y_test)}")
        
        print("✅ Data splitting completed!")
    
    def scale_features(self):
        """Scale numerical features"""
        print("\n" + "=" * 80)
        print("📏 STEP 10: FEATURE SCALING")
        print("=" * 80)
        
        # Use RobustScaler for better handling of outliers
        self.scaler = RobustScaler()
        
        # Scale training data
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("✅ Feature scaling completed using RobustScaler!")
        print(f"📊 Scaled training set shape: {self.X_train_scaled.shape}")
        print(f"📊 Scaled test set shape: {self.X_test_scaled.shape}")
    
    def train_model(self):
        """Train the Random Forest model with hyperparameter tuning"""
        print("\n" + "=" * 80)
        print("🤖 STEP 11: MODEL TRAINING")
        print("=" * 80)
        
        # Define parameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [400, 500, 600],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        print("🔍 Performing GridSearchCV for hyperparameter tuning...")
        
        # Initialize Random Forest
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Perform GridSearchCV
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        
        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"✅ Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return self.model
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("\n" + "=" * 80)
        print("📊 STEP 12: MODEL EVALUATION")
        print("=" * 80)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test_scaled)
        y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print("📊 Performance Metrics:")
        print(f"   Accuracy:  {accuracy * 100:.2f}%")
        print(f"   Precision: {precision * 100:.2f}%")
        print(f"   Recall:    {recall * 100:.2f}%")
        print(f"   F1-Score:  {f1 * 100:.2f}%")
        print(f"   AUC-ROC:   {auc * 100:.2f}%")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\n📊 Confusion Matrix:")
        print(cm)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🎯 Top 10 Feature Importance:")
        print(feature_importance.head(10))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'feature_importance': feature_importance
        }
    
    def save_model_and_preprocessors(self, accuracy):
        """Save model and preprocessors"""
        print("\n" + "=" * 80)
        print("💾 STEP 13: SAVING MODEL AND PREPROCESSORS")
        print("=" * 80)
        
        # Create filename with accuracy
        model_filename = f"models/random_forest_model_{accuracy*100:.2f}%.pkl"
        scaler_filename = f"models/scaler_{accuracy*100:.2f}%.pkl"
        encoder_filename = f"models/encoder_{accuracy*100:.2f}%.pkl"
        feature_selector_filename = f"models/feature_selector_{accuracy*100:.2f}%.pkl"
        
        # Save model and preprocessors
        joblib.dump(self.model, model_filename)
        joblib.dump(self.scaler, scaler_filename)
        joblib.dump(self.encoder, encoder_filename)
        joblib.dump(self.feature_selector, feature_selector_filename)
        
        print(f"✅ Model saved as: {model_filename}")
        print(f"✅ Scaler saved as: {scaler_filename}")
        print(f"✅ Encoder saved as: {encoder_filename}")
        print(f"✅ Feature selector saved as: {feature_selector_filename}")
        
        return model_filename
    
    def run_complete_pipeline(self, filepath):
        """Run the complete preprocessing and training pipeline"""
        start_time = time.time()
        
        # Execute all steps
        self.load_data(filepath)
        self.explore_target_distribution()
        self.handle_missing_values()
        self.remove_outliers_and_anomalies()
        self.feature_engineering()
        self.encode_categorical_variables()
        self.feature_selection()
        self.apply_smote_balancing()
        self.split_data()
        self.scale_features()
        self.train_model()
        results = self.evaluate_model()
        model_filename = self.save_model_and_preprocessors(results['accuracy'])
        
        end_time = time.time()
        
        print("\n" + "=" * 80)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"⏱️  Total execution time: {end_time - start_time:.2f} seconds")
        print(f"📁 Model saved as: {model_filename}")
        print("=" * 80)
        
        return results, model_filename

if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = StrokeDataPreprocessor()
    
    # Run complete pipeline
    results, model_filename = preprocessor.run_complete_pipeline("data/healthcare-dataset-stroke-data.csv") 