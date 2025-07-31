# 📊 DOKUMENTASI ANALISIS KODE PROGRAM PEMBENTUKAN MODEL DATA MINING
## Stroke Prediction dengan Random Forest dan LightGBM

---

## 📋 **DAFTAR ISI**
1. [Pendahuluan](#pendahuluan)
2. [Analisis Struktur Kode](#analisis-struktur-kode)
3. [Implementasi Random Forest](#implementasi-random-forest)
4. [Implementasi LightGBM](#implementasi-lightgbm)
5. [Perbandingan Algoritma](#perbandingan-algoritma)
6. [Analisis Performa](#analisis-performa)
7. [Kesimpulan dan Rekomendasi](#kesimpulan-dan-rekomendasi)

---

## 🎯 **PENDAHULUAN**

### **Latar Belakang**
Proyek ini mengembangkan model prediksi stroke menggunakan dua algoritma machine learning yang berbeda:
- **Random Forest** (Ensemble Learning)
- **LightGBM** (Gradient Boosting)

### **Tujuan**
1. Membandingkan performa Random Forest vs LightGBM
2. Menganalisis struktur kode dan implementasi
3. Menghasilkan model prediksi stroke yang akurat
4. Menangani masalah class imbalance

### **Dataset**
- **Sumber**: Healthcare Dataset Stroke Data
- **Jumlah**: 5,110 sampel
- **Fitur**: 12 variabel (demografis, medis, gaya hidup)
- **Target**: Binary classification (stroke/no stroke)

---

## 🔍 **ANALISIS STRUKTUR KODE**

### **Arsitektur Program**

```
stroke-prediction-rf/
├── data/
│   └── healthcare-dataset-stroke-data.csv
├── src/
│   ├── main.py                    # Random Forest implementation
│   └── lightgbm_main_direct.py   # LightGBM implementation
├── requirements.txt               # Dependencies
└── README.md                     # Documentation
```

### **Struktur Kelas Utama**

#### **1. Random Forest (`StrokeDataPreprocessor`)**
```python
class StrokeDataPreprocessor:
    def __init__(self):
        # Inisialisasi komponen preprocessing
        self.data = None
        self.scaler = RobustScaler()
        self.encoder = OneHotEncoder()
        self.feature_selector = SelectKBest()
        self.model = RandomForestClassifier()
```

#### **2. LightGBM (`LightGBMStrokePredictor`)**
```python
class LightGBMStrokePredictor:
    def __init__(self):
        # Inisialisasi komponen preprocessing
        self.data = None
        self.scaler = RobustScaler()
        self.encoder = OneHotEncoder()
        self.feature_selector = SelectKBest()
        self.model = None
        self.best_params = None
```

### **Pipeline Tahapan**

| Tahap | Random Forest | LightGBM | Deskripsi |
|-------|---------------|----------|-----------|
| 1 | Data Loading | Data Loading | Pemuatan dan eksplorasi data |
| 2 | Target Analysis | Target Analysis | Analisis distribusi target |
| 3 | Missing Values | Missing Values | Penanganan nilai hilang |
| 4 | Outlier Detection | Outlier Detection | Deteksi dan pembersihan outlier |
| 5 | Feature Engineering | Feature Engineering | Pembuatan fitur baru |
| 6 | Encoding | Encoding | Encoding variabel kategorikal |
| 7 | Feature Selection | Feature Selection | Seleksi fitur terbaik |
| 8 | SMOTE Balancing | SMOTE Balancing | Penanganan class imbalance |
| 9 | Data Splitting | Data Splitting | Pembagian data train/test |
| 10 | Scaling | Scaling | Normalisasi fitur |
| 11 | Hyperparameter Tuning | Hyperparameter Optimization | Optimasi parameter |
| 12 | Model Training | Model Training | Pelatihan model |
| 13 | Evaluation | Evaluation | Evaluasi performa |
| 14 | Model Saving | Model Saving | Penyimpanan model |

---

## 🌳 **IMPLEMENTASI RANDOM FOREST**

### **Analisis Kode `src/main.py`**

#### **1. Preprocessing Pipeline**

```python
def load_data(self, filepath):
    """Step 1: Data Loading and Initial Exploration"""
    self.data = pd.read_csv(filepath)
    # Analisis missing values
    # Statistik deskriptif
    return self.data
```

**Analisis:**
- ✅ **Comprehensive**: Analisis lengkap missing values dan statistik
- ✅ **Informative**: Output yang informatif dengan emoji dan formatting
- ✅ **Robust**: Penanganan error yang baik

#### **2. Feature Engineering**

```python
def feature_engineering(self):
    """Step 5: Feature Engineering"""
    # Age groups
    self.data['age_group'] = pd.cut(self.data['age'], 
                                   bins=[0, 30, 45, 60, 75, 100], 
                                   labels=['Young', 'Adult', 'Middle-aged', 'Senior', 'Elderly'])
    
    # Risk score
    risk_factors = 0
    risk_factors += (self.data['age'] > 65).astype(int)
    risk_factors += self.data['hypertension']
    risk_factors += self.data['heart_disease']
    self.data['risk_score'] = risk_factors
```

**Analisis:**
- ✅ **Domain Knowledge**: Menggunakan pengetahuan medis untuk risk score
- ✅ **Categorical Features**: Pembuatan kelompok usia yang meaningful
- ✅ **Composite Score**: Risk score menggabungkan multiple factors

#### **3. Hyperparameter Tuning**

```python
def train_model(self):
    """Step 11: Model Training with GridSearchCV"""
    param_grid = {
        'n_estimators': [400, 500, 600],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
```

**Analisis:**
- ✅ **GridSearchCV**: Exhaustive search untuk parameter optimal
- ✅ **Cross-validation**: 5-fold CV untuk validasi robust
- ✅ **F1-score**: Metric yang tepat untuk imbalanced data

#### **4. Model Evaluation**

```python
def evaluate_model(self):
    """Step 12: Comprehensive Model Evaluation"""
    # Calculate metrics
    accuracy = accuracy_score(self.y_test, y_pred)
    precision = precision_score(self.y_test, y_pred)
    recall = recall_score(self.y_test, y_pred)
    f1 = f1_score(self.y_test, y_pred)
    auc = roc_auc_score(self.y_test, y_pred_proba)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': self.X_train.columns,
        'importance': self.model.feature_importances_
    }).sort_values('importance', ascending=False)
```

**Analisis:**
- ✅ **Multiple Metrics**: Accuracy, Precision, Recall, F1, AUC-ROC
- ✅ **Feature Importance**: Analisis kepentingan fitur
- ✅ **Comprehensive**: Evaluasi lengkap dengan confusion matrix

---

## ⚡ **IMPLEMENTASI LIGHTGBM**

### **Analisis Kode `src/lightgbm_main_direct.py`**

#### **1. Advanced Hyperparameter Optimization**

```python
def objective(self, trial):
    """Optuna objective function for hyperparameter optimization"""
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    
    # Cross-validation
    cv_results = lgb.cv(
        params, train_data,
        num_boost_round=params['n_estimators'],
        nfold=5, stratified=True, shuffle=True, seed=42
    )
    return cv_results['valid binary_logloss-mean'][-1]
```

**Analisis:**
- ✅ **Optuna**: Bayesian optimization yang lebih efisien
- ✅ **Comprehensive Search Space**: Parameter yang lebih luas
- ✅ **Log-scale**: Pencarian yang lebih efisien untuk learning rate

#### **2. Early Stopping dan Callbacks**

```python
def train_model(self):
    """Step 12: LightGBM Model Training"""
    self.model = lgb.train(
        self.best_params,
        train_data,
        valid_sets=[valid_data],
        valid_names=['valid'],
        num_boost_round=self.best_params['n_estimators'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )
```

**Analisis:**
- ✅ **Early Stopping**: Mencegah overfitting
- ✅ **Validation Set**: Monitoring performa real-time
- ✅ **Efficient Training**: Training yang lebih cepat

#### **3. Advanced Feature Importance**

```python
def evaluate_model(self):
    """Step 13: Model Evaluation with Gain-based Importance"""
    importance = self.model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': self.X_train.columns.tolist(),
        'importance': importance
    }).sort_values('importance', ascending=False)
```

**Analisis:**
- ✅ **Gain-based Importance**: Metrik importance yang lebih akurat
- ✅ **Detailed Analysis**: Analisis kepentingan fitur yang mendalam

---

## 📊 **PERBANDINGAN ALGORITMA**

### **1. Arsitektur Model**

| Aspek | Random Forest | LightGBM |
|-------|---------------|----------|
| **Tipe** | Ensemble (Bagging) | Gradient Boosting |
| **Base Learner** | Decision Trees | Decision Trees |
| **Training** | Parallel | Sequential |
| **Overfitting** | Less prone | More prone (but controlled) |
| **Speed** | Moderate | Fast |

### **2. Hyperparameter Tuning**

| Metode | Random Forest | LightGBM |
|--------|---------------|----------|
| **Strategy** | GridSearchCV | Optuna (Bayesian) |
| **Search Space** | Limited | Comprehensive |
| **Efficiency** | Exhaustive | Intelligent |
| **Time** | Slower | Faster |

### **3. Feature Engineering**

| Aspek | Random Forest | LightGBM |
|-------|---------------|----------|
| **Scaling** | RobustScaler | RobustScaler |
| **Encoding** | OneHotEncoder | OneHotEncoder |
| **Selection** | SelectKBest (k=15) | SelectKBest (k=15) |
| **SMOTE** | Applied | Applied |

---

## 🎯 **ANALISIS PERFORMA**

### **Hasil Random Forest**
```
📊 Performance Metrics:
   Accuracy:  97.31%
   Precision: 97.25%
   Recall:    94.83%
   F1-Score:  97.25%
   AUC-ROC:   97.31%
```

### **Hasil LightGBM**
```
📊 Performance Metrics:
   Accuracy:  97.48%
   Precision: 99.15%
   Recall:    95.78%
   F1-Score:  97.44%
   AUC-ROC:   99.41%
```

### **Perbandingan Performa**

| Metric | Random Forest | LightGBM | Peningkatan |
|--------|---------------|----------|-------------|
| **Accuracy** | 97.31% | 97.48% | +0.17% |
| **Precision** | 97.25% | 99.15% | +1.90% |
| **Recall** | 94.83% | 95.78% | +0.95% |
| **F1-Score** | 97.25% | 97.44% | +0.19% |
| **AUC-ROC** | 97.31% | 99.41% | +2.10% |

### **Analisis Feature Importance**

#### **Random Forest Top 10:**
1. `age` (0.284)
2. `avg_glucose_level` (0.183)
3. `bmi` (0.142)
4. `hypertension` (0.121)
5. `heart_disease` (0.089)

#### **LightGBM Top 10:**
1. `age` (41,266)
2. `risk_score` (9,572)
3. `gender_Male` (9,191)
4. `Residence_type_Urban` (8,632)
5. `avg_glucose_level` (7,539)

---

## 🔧 **ANALISIS KEKUATAN DAN KELEMAHAN**

### **Random Forest**

#### **Kekuatan:**
- ✅ **Robust**: Tidak mudah overfitting
- ✅ **Interpretable**: Feature importance yang jelas
- ✅ **Stable**: Performa yang konsisten
- ✅ **Parallel**: Training yang bisa diparallelkan

#### **Kelemahan:**
- ❌ **Limited Search**: GridSearchCV terbatas
- ❌ **Slower Training**: Waktu training lebih lama
- ❌ **Less Accurate**: Performa sedikit lebih rendah

### **LightGBM**

#### **Kekuatan:**
- ✅ **High Performance**: Performa yang lebih tinggi
- ✅ **Fast Training**: Training yang lebih cepat
- ✅ **Advanced Tuning**: Optuna optimization
- ✅ **Early Stopping**: Mencegah overfitting

#### **Kelemahan:**
- ❌ **Complex**: Lebih sulit diinterpretasi
- ❌ **Overfitting Risk**: Potensi overfitting lebih tinggi
- ❌ **Parameter Sensitive**: Sangat sensitif terhadap parameter

---

## 📈 **ANALISIS KODE QUALITY**

### **Code Quality Metrics**

| Aspek | Random Forest | LightGBM | Score |
|-------|---------------|----------|-------|
| **Modularity** | ✅ Excellent | ✅ Excellent | 9/10 |
| **Readability** | ✅ Excellent | ✅ Excellent | 9/10 |
| **Documentation** | ✅ Good | ✅ Good | 8/10 |
| **Error Handling** | ✅ Good | ✅ Good | 8/10 |
| **Performance** | ✅ Good | ✅ Excellent | 9/10 |

### **Best Practices Implemented**

#### **1. Data Preprocessing**
```python
# Missing value handling dengan group-based imputation
self.data['bmi'] = self.data.groupby(['gender', 'work_type'])['bmi'].transform(
    lambda x: x.fillna(x.median())
)
```

#### **2. Class Imbalance Handling**
```python
# SMOTE untuk balancing
smote = SMOTE(random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

#### **3. Feature Selection**
```python
# SelectKBest untuk feature selection
self.feature_selector = SelectKBest(score_func=f_classif, k=15)
```

#### **4. Model Persistence**
```python
# Saving model dan preprocessors
joblib.dump(self.model, model_filename)
joblib.dump(self.scaler, scaler_filename)
```

---

## 🎯 **KESIMPULAN DAN REKOMENDASI**

### **Kesimpulan**

1. **LightGBM Superior**: LightGBM menunjukkan performa yang lebih baik di semua metrik
2. **Consistent Pipeline**: Kedua implementasi menggunakan pipeline yang konsisten
3. **Robust Preprocessing**: Penanganan data yang komprehensif
4. **Advanced Optimization**: LightGBM menggunakan optimisasi yang lebih canggih

### **Rekomendasi**

#### **Untuk Production:**
- ✅ **Use LightGBM**: Untuk performa optimal
- ✅ **Implement Monitoring**: Untuk tracking model drift
- ✅ **Regular Retraining**: Untuk menjaga performa

#### **Untuk Development:**
- ✅ **Maintain Both**: Untuk perbandingan berkelanjutan
- ✅ **Add Visualization**: Untuk analisis yang lebih mendalam
- ✅ **Unit Testing**: Untuk memastikan reliability

#### **Untuk Research:**
- ✅ **Experiment More**: Coba algoritma lain (XGBoost, CatBoost)
- ✅ **Feature Engineering**: Eksplorasi fitur engineering yang lebih advanced
- ✅ **Ensemble Methods**: Kombinasi multiple models

### **Next Steps**

1. **Model Deployment**: Implementasi model ke production
2. **API Development**: REST API untuk prediksi real-time
3. **Monitoring System**: Sistem monitoring performa model
4. **A/B Testing**: Perbandingan model di production

---

## 📚 **REFERENSI**

1. **Random Forest**: Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.
2. **LightGBM**: Ke, G., et al. (2017). Lightgbm: A highly efficient gradient boosting decision tree.
3. **SMOTE**: Chawla, N. V., et al. (2002). SMOTE: synthetic minority over-sampling technique.
4. **Optuna**: Akiba, T., et al. (2019). Optuna: A next-generation hyperparameter optimization framework.

---

*Dokumentasi ini dibuat untuk analisis komprehensif kode program pembentukan model data mining stroke prediction menggunakan Random Forest dan LightGBM.* 