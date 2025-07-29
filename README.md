# Stroke Prediction with Random Forest and SMOTE

This project implements a stroke prediction model using Random Forest classifier with SMOTE (Synthetic Minority Over-sampling Technique) for handling imbalanced data. The model achieves 97.38% accuracy and is deployed as a beautiful Streamlit web application.

## Features

- **High Accuracy**: 97.38% accuracy with balanced precision and recall
- **SMOTE Integration**: Handles imbalanced stroke data effectively
- **Beautiful Web Interface**: Modern Streamlit app with interactive features
- **Height & Weight Input**: Automatic BMI calculation from height and weight
- **Risk Factor Analysis**: Identifies and explains risk factors
- **Visual Results**: Interactive gauges and charts for prediction results

## Project Structure

```
stroke-prediction-rf/
├── data/
│   └── healthcare-dataset-stroke-data.csv
├── src/
│   ├── main.py              # Training script with preprocessing display
│   └── utils.py             # Utility functions
├── app.py                   # Streamlit web application
├── random_forest_model_97.38%.pkl  # Trained model
├── requirements.txt         # Python dependencies
├── setup_venv.sh           # Virtual environment setup
├── run_project.sh          # Training script runner
├── run_app.sh              # Streamlit app runner
└── README.md               # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd stroke-prediction-rf
   ```

2. **Set up virtual environment**:
   ```bash
   chmod +x setup_venv.sh
   ./setup_venv.sh
   ```

3. **Install dependencies**:
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the model and see preprocessing steps:

```bash
chmod +x run_project.sh
./run_project.sh
```

This will:
- Load and preprocess the dataset
- Display detailed preprocessing information
- Train the Random Forest model with SMOTE
- Show evaluation metrics
- Save the trained model

### Running the Web Application

To launch the Streamlit web app:

```bash
chmod +x run_app.sh
./run_app.sh
```

Or manually:
```bash
source venv/bin/activate
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Web Application Features

### 🏠 Home Page
- Overview of the application
- Model performance metrics
- Risk factors analyzed

### 📊 Prediction Page
- **Personal Information**: Age, gender, marital status, work type, residence, smoking status
- **Medical Information**: Hypertension, heart disease, glucose level
- **Body Measurements**: Height and weight (automatic BMI calculation)
- **Interactive Results**: 
  - Risk probability gauge
  - Color-coded risk assessment
  - Risk factor analysis
  - Professional medical advice

### 📈 Model Information
- Technical model details
- Performance metrics
- Feature importance visualization

### ℹ️ About
- Application purpose and disclaimers
- Technical information
- Data sources

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 97.38% |
| Precision | 96.15% |
| Recall | 98.46% |
| F1-Score | 97.29% |

## Key Features

### Data Preprocessing
- Missing BMI imputation using median by gender and work type
- One-hot encoding for categorical variables
- SMOTE for handling class imbalance
- Feature scaling and normalization

### Model Architecture
- **Algorithm**: Random Forest Classifier
- **Estimators**: 600 trees
- **Random State**: 42 (for reproducibility)
- **Data Balancing**: SMOTE technique

### Web Interface
- **Responsive Design**: Works on desktop and mobile
- **Interactive Forms**: Easy data input with validation
- **Visual Results**: Gauge charts and color-coded risk assessment
- **Risk Analysis**: Automatic identification of risk factors
- **Professional UI**: Modern design with medical disclaimers

## Input Parameters

The model accepts the following parameters:

1. **Personal Information**:
   - Age (1-120 years)
   - Gender (Female/Male)
   - Marital Status (Yes/No)
   - Work Type (Private/Self-employed/Govt_job/Never_worked)
   - Residence Type (Urban/Rural)
   - Smoking Status (never smoked/formerly smoked/smokes/Unknown)

2. **Medical Information**:
   - Hypertension (Yes/No)
   - Heart Disease (Yes/No)
   - Average Glucose Level (50-300 mg/dL)

3. **Body Measurements**:
   - Height (100-250 cm)
   - Weight (30-200 kg)
   - **Automatic BMI calculation** with category classification

## Risk Assessment

The application provides:

- **Risk Probability**: Percentage-based stroke risk assessment
- **Risk Level**: Low/High risk classification
- **Risk Factors**: Automatic identification of contributing factors
- **Medical Advice**: Appropriate recommendations based on risk level

## Important Disclaimer

⚠️ **This application is for educational and screening purposes only. It should not be used as a substitute for professional medical diagnosis, treatment, or advice. Always consult with qualified healthcare professionals for medical concerns.**

## Technical Requirements

- Python 3.8+
- Virtual environment (recommended)
- Internet connection (for first-time package installation)

## Dependencies

- `numpy==1.24.3`
- `pandas==2.0.3`
- `matplotlib==3.7.2`
- `seaborn==0.12.2`
- `scikit-learn==1.3.0`
- `imbalanced-learn==0.11.0`
- `joblib==1.3.2`
- `streamlit==1.28.1`
- `plotly==5.17.0`

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is for educational purposes.