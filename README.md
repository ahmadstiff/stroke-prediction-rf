# 🧠 Stroke Prediction Analysis Dashboard

A comprehensive machine learning project for stroke prediction using Random Forest algorithm with advanced preprocessing and feature engineering.

## 📊 Project Overview

This project implements a stroke prediction model with the following key features:

- **Algorithm**: Random Forest Classifier
- **Accuracy**: 97.79%
- **AUC-ROC**: 99.58%
- **Precision**: 99.47%
- **Recall**: 96.09%

## 🏗️ Project Structure

```
stroke-prediction-rf/
├── app.py                          # Streamlit web application
├── src/
│   ├── main.py                     # Main training pipeline
│   ├── utils.py                    # Utility functions
│   └── lightgbm_main_direct.py    # LightGBM implementation (reference)
├── models/                         # Trained models and preprocessors
│   ├── random_forest_model_97.79%.pkl
│   ├── scaler_97.79%.pkl
│   ├── encoder_97.79%.pkl
│   └── feature_selector_97.79%.pkl
├── data/
│   └── healthcare-dataset-stroke-data.csv
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 🔧 Model Features

### Enhanced Risk Score Calculation (0-12 points):
- **Age > 65**: 2 points
- **Age > 75**: 1 additional point
- **Hypertension**: 2 points
- **Heart Disease**: 2 points
- **Glucose > 140**: 1 point
- **Glucose > 200**: 1 additional point
- **BMI > 30**: 1 point
- **BMI > 40**: 1 additional point
- **Smoking**: 1 point

### Risk Assessment Thresholds:
- **🔴 VERY HIGH RISK**: ≥50%
- **🟠 HIGH RISK**: ≥35%
- **🟡 MODERATE RISK**: ≥20%
- **🟢 LOW RISK**: ≥10%

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 97.79% |
| Precision | 99.47% |
| Recall | 96.09% |
| F1-Score | 97.75% |
| AUC-ROC | 99.58% |

## 🎯 Key Features

### 1. Advanced Preprocessing Pipeline
- Missing value imputation with group-based approach
- Outlier detection and handling
- Feature engineering with risk score calculation
- SMOTE balancing for class imbalance

### 2. Feature Selection
- SelectKBest with f_classif (k=15)
- Optimized feature set for maximum performance

### 3. Hyperparameter Tuning
- GridSearchCV for Random Forest optimization
- Cross-validation for robust evaluation

### 4. Interactive Web Interface
- Real-time stroke risk prediction
- Detailed risk factor analysis
- Visual performance metrics
- Comprehensive documentation

## 🔍 Dataset Information

- **Source**: Healthcare Dataset Stroke Data
- **Size**: 5,110 samples
- **Features**: 12 variables (demographic, medical, lifestyle)
- **Target**: Binary classification (stroke/no stroke)
- **Class Imbalance**: 95.13% no stroke, 4.87% stroke

## 📋 Usage

1. **Navigate to Prediction Page**: Click "🔮 Predict Stroke" in the sidebar
2. **Enter Patient Data**: Fill in all required fields
3. **Get Prediction**: Click "🔮 Predict Stroke Risk" button
4. **Review Results**: View risk assessment and detailed analysis

## 🛠️ Technical Stack

- **Python 3.8+**
- **scikit-learn**: Machine learning algorithms
- **pandas, numpy**: Data manipulation
- **streamlit**: Web application framework
- **plotly**: Interactive visualizations
- **imbalanced-learn**: SMOTE balancing
- **joblib**: Model persistence

## 📚 Documentation

- `ANALISIS_KODE_PROGRAM.md`: Comprehensive code analysis
- `ANALISIS_TEKNIS_DETAIL.md`: Technical deep dive
- `RINGKASAN_EKSEKUTIF.md`: Executive summary
- `PROJECT_SUMMARY.md`: Project overview

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

---

**Note**: This model is for educational and research purposes. For medical diagnosis, always consult healthcare professionals.