# Stroke Prediction - Random Forest vs LightGBM

This project implements stroke prediction models using Random Forest and LightGBM algorithms with advanced preprocessing techniques.

## 🎯 **Project Objectives**

1. **Menghasilkan model klasifikasi risiko penyakit stroke** menggunakan algoritma Random Forest dan LightGBM berdasarkan dataset yang tersedia
2. **Menganalisis dan membandingkan kinerja algoritma** Random Forest dan LightGBM untuk menentukan algoritma dengan akurasi terbaik

## 🏆 **Results Summary**

### **Random Forest Performance:**
- **Accuracy**: 97.33%
- **Precision**: 99.78%
- **Recall**: 94.83%
- **F1-Score**: 97.25%
- **AUC-ROC**: 97.31%

### **LightGBM Performance:**
- **Accuracy**: 97.49%
- **Precision**: 99.15%
- **Recall**: 95.88%
- **F1-Score**: 97.49%
- **AUC-ROC**: 99.43%

## 🏆 **Best Algorithm: LightGBM**

**LightGBM achieves better overall performance** with higher accuracy and AUC-ROC score.

## 📁 **Project Structure**

```
stroke-prediction-rf/
├── data/
│   └── healthcare-dataset-stroke-data.csv
├── src/
│   ├── main.py                    # Clean Random Forest implementation
│   ├── advanced_preprocessing.py  # Advanced preprocessing pipeline
│   └── utils.py                   # Utility functions
├── requirements.txt               # Python dependencies
├── run_lightgbm_pipeline.py      # LightGBM pipeline
└── README.md                     # This file
```

## 🚀 **Usage**

### **Installation:**
```bash
pip install -r requirements.txt
```

### **Run Random Forest Pipeline:**
```bash
python src/main.py
```

### **Run LightGBM Pipeline:**
```bash
python run_lightgbm_pipeline.py
```

## 📈 **Model Performance Comparison**

| **Metric** | **Random Forest** | **LightGBM** | **Winner** |
|------------|-------------------|--------------|------------|
| **Accuracy** | 97.33% | **97.49%** | ✅ LightGBM |
| **Precision** | **99.78%** | 99.15% | ✅ Random Forest |
| **Recall** | 94.83% | **95.88%** | ✅ LightGBM |
| **F1-Score** | 97.25% | **97.49%** | ✅ LightGBM |
| **AUC-ROC** | 97.31% | **99.43%** | ✅ LightGBM |

## 🎯 **Key Findings**

### **1. LightGBM Superiority:**
- Higher accuracy (97.49% vs 97.33%)
- Better recall (95.88% vs 94.83%)
- Higher F1-score (97.49% vs 97.25%)
- Much better AUC-ROC (99.43% vs 97.31%)

### **2. Random Forest Strengths:**
- Higher precision (99.78% vs 99.15%)
- More interpretable feature importance
- Better stability with advanced preprocessing

## 📋 **Requirements**

- Python 3.8+
- numpy
- pandas
- scikit-learn
- imbalanced-learn
- joblib
- lightgbm
- optuna
- streamlit
- plotly

## ⚠️ **Important Notes**

### **Medical Disclaimer:**
This application is for educational and screening purposes only. It should not replace professional medical diagnosis, treatment, or advice. Always consult with qualified healthcare professionals for medical concerns.

### **Model Limitations:**
- Based on historical healthcare data
- May not capture all individual risk factors
- Should be used as a screening tool only

## 🏆 **Conclusion**

**LightGBM is the best algorithm** for stroke prediction with:
- ✅ Higher accuracy (97.49%)
- ✅ Better recall and F1-score
- ✅ Superior AUC-ROC (99.43%)
- ✅ Advanced hyperparameter optimization

**Recommendation:** Use LightGBM for production implementation due to superior overall performance in stroke risk prediction.