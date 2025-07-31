import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Stroke Prediction Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Dark theme background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main content background */
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Header styling */
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        font-weight: bold;
    }
    
    /* Metric cards with glass effect */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(255, 255, 255, 0.7) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin: 0.5rem;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    .success-metric {
        border-left-color: #28a745;
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.1) 0%, rgba(255, 255, 255, 0.9) 100%);
    }
    
    .warning-metric {
        border-left-color: #ffc107;
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 255, 255, 0.9) 100%);
    }
    
    .info-metric {
        border-left-color: #17a2b8;
        background: linear-gradient(135deg, rgba(23, 162, 184, 0.1) 0%, rgba(255, 255, 255, 0.9) 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    /* Text styling */
    h1, h2, h3 {
        color: #2c3e50 !important;
    }
    
    /* Table styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 10px !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Stroke Prediction Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔮 Predict Stroke", "📈 Model Performance", "🔍 Feature Analysis", "🤖 Model Comparison", "📋 Documentation"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Predict Stroke":
        show_prediction_page()
    elif page == "📈 Model Performance":
        show_performance_page()
    elif page == "🔍 Feature Analysis":
        show_feature_analysis_page()
    elif page == "🤖 Model Comparison":
        show_model_comparison_page()
    elif page == "📋 Documentation":
        show_documentation_page()

def show_home_page():
    st.markdown("## 🎯 Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Project Summary
        This project implements stroke prediction models using two different machine learning algorithms:
        
        - **Random Forest** (Ensemble Learning)
        - **LightGBM** (Gradient Boosting)
        
        ### 🎯 Key Objectives
        - Compare Random Forest vs LightGBM performance
        - Analyze code structure and implementation
        - Generate accurate stroke prediction models
        - Handle class imbalance issues
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Dataset Information
        - **Source**: Healthcare Dataset Stroke Data
        - **Size**: 5,110 samples
        - **Features**: 12 variables (demographic, medical, lifestyle)
        - **Target**: Binary classification (stroke/no stroke)
        - **Class Imbalance**: 95.13% no stroke, 4.87% stroke
        """)
    
    # Performance Metrics Cards
    st.markdown("## 🏆 Model Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card success-metric">
            <h3>Random Forest</h3>
            <p><strong>Accuracy:</strong> 97.79%</p>
            <p><strong>AUC-ROC:</strong> 99.58%</p>
            <p><strong>Precision:</strong> 99.47%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card info-metric">
            <h3>LightGBM</h3>
            <p><strong>Accuracy:</strong> 97.48%</p>
            <p><strong>AUC-ROC:</strong> 99.41%</p>
            <p><strong>Precision:</strong> 99.15%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card warning-metric">
            <h3>Winner</h3>
            <p><strong>Algorithm:</strong> Random Forest</p>
            <p><strong>Improvement:</strong> +0.26%</p>
            <p><strong>Status:</strong> Production Ready</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card info-metric">
            <h3>Pipeline</h3>
            <p><strong>Steps:</strong> 14</p>
            <p><strong>Features:</strong> 15 selected</p>
            <p><strong>Balancing:</strong> SMOTE</p>
        </div>
        """, unsafe_allow_html=True)

def show_performance_page():
    st.markdown("## 📈 Model Performance Analysis")
    
    # Performance comparison data
    performance_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
        'Random Forest': [97.74, 99.57, 95.88, 97.69, 99.61],
        'LightGBM': [97.48, 99.15, 95.78, 97.44, 99.41],
        'Improvement': [0.26, 0.42, 0.10, 0.25, 0.20]
    }
    
    df_performance = pd.DataFrame(performance_data)
    
    # Performance comparison chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Random Forest',
        x=df_performance['Metric'],
        y=df_performance['Random Forest'],
        marker_color='#1f77b4',
        text=df_performance['Random Forest'],
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        name='LightGBM',
        x=df_performance['Metric'],
        y=df_performance['LightGBM'],
        marker_color='#ff7f0e',
        text=df_performance['LightGBM'],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Metrics',
        yaxis_title='Percentage (%)',
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.markdown("### 📊 Detailed Metrics")
    st.dataframe(df_performance, use_container_width=True)
    
    # Confusion Matrix
    st.markdown("### 🎯 Random Forest Confusion Matrix")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion matrix data
        cm_data = np.array([[968, 4], [40, 932]])
        labels = ['No Stroke', 'Stroke']
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=cm_data,
            texttemplate="%{text}",
            textfont={"size": 16},
            showscale=True
        ))
        
        fig_cm.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=400
        )
        
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        st.markdown("""
        ### 📋 Classification Report
        
        **Random Forest Results:**
        - **True Negatives:** 968 (No Stroke correctly predicted)
        - **False Positives:** 4 (No Stroke predicted as Stroke)
        - **False Negatives:** 40 (Stroke predicted as No Stroke)
        - **True Positives:** 932 (Stroke correctly predicted)
        
        **Key Insights:**
        - Very low false positive rate (0.41%)
        - Good recall for stroke detection (95.88%)
        - High precision for stroke prediction (99.57%)
        """)

def show_feature_analysis_page():
    st.markdown("## 🔍 Feature Importance Analysis")
    
    # Feature importance data
    rf_features = {
        'Feature': ['age', 'risk_score', 'Residence_type_Urban', 'smoking_status_formerly smoked', 
                   'hypertension', 'avg_glucose_level', 'ever_married_Yes', 'gender_Male', 
                   'work_type_Private', 'bmi'],
        'Importance': [21.50, 12.23, 7.41, 7.36, 7.21, 6.77, 6.58, 6.08, 5.69, 5.59]
    }
    
    lgbm_features = {
        'Feature': ['age', 'risk_score', 'gender_Male', 'Residence_type_Urban', 
                   'avg_glucose_level', 'smoking_status_formerly smoked', 'hypertension', 
                   'work_type_Private', 'ever_married_Yes', 'bmi'],
        'Importance': [41.3, 9.6, 9.2, 8.6, 7.5, 6.4, 6.1, 5.8, 5.5, 5.2]
    }
    
    df_rf = pd.DataFrame(rf_features)
    df_lgbm = pd.DataFrame(lgbm_features)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌳 Random Forest Feature Importance")
        
        fig_rf = px.bar(df_rf.head(10), x='Importance', y='Feature', 
                       orientation='h', color='Importance',
                       color_continuous_scale='Blues',
                       title='Top 10 Features - Random Forest')
        
        fig_rf.update_layout(height=500)
        st.plotly_chart(fig_rf, use_container_width=True)
    
    with col2:
        st.markdown("### ⚡ LightGBM Feature Importance")
        
        fig_lgbm = px.bar(df_lgbm.head(10), x='Importance', y='Feature', 
                          orientation='h', color='Importance',
                          color_continuous_scale='Oranges',
                          title='Top 10 Features - LightGBM')
        
        fig_lgbm.update_layout(height=500)
        st.plotly_chart(fig_lgbm, use_container_width=True)
    
    # Feature comparison
    st.markdown("### 📊 Feature Importance Comparison")
    
    # Create comparison chart
    comparison_data = pd.DataFrame({
        'Feature': df_rf['Feature'],
        'Random Forest': df_rf['Importance'],
        'LightGBM': df_lgbm['Importance']
    })
    
    fig_comp = go.Figure()
    
    fig_comp.add_trace(go.Bar(
        name='Random Forest',
        x=comparison_data['Feature'],
        y=comparison_data['Random Forest'],
        marker_color='#1f77b4'
    ))
    
    fig_comp.add_trace(go.Bar(
        name='LightGBM',
        x=comparison_data['Feature'],
        y=comparison_data['LightGBM'],
        marker_color='#ff7f0e'
    ))
    
    fig_comp.update_layout(
        title='Feature Importance Comparison',
        xaxis_title='Features',
        yaxis_title='Importance (%)',
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Key insights
    st.markdown("### 💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Random Forest Insights:**
        - More balanced feature importance distribution
        - Age is the most important feature (21.50%)
        - Risk score shows high importance (12.23%)
        - Multiple features contribute significantly
        """)
    
    with col2:
        st.markdown("""
        **LightGBM Insights:**
        - Age dominates with 41.3% importance
        - More focused on fewer key features
        - Risk score less prominent (9.6%)
        - Gender shows higher importance (9.2%)
        """)

def show_model_comparison_page():
    st.markdown("## 🤖 Model Comparison Analysis")
    
    # Model characteristics comparison
    comparison_data = {
        'Aspect': ['Algorithm Type', 'Training Method', 'Hyperparameter Tuning', 
                  'Feature Importance', 'Overfitting Risk', 'Training Speed', 
                  'Interpretability', 'Production Readiness'],
        'Random Forest': ['Ensemble (Bagging)', 'Parallel', 'GridSearchCV', 
                        'Balanced', 'Low', 'Moderate', 'High', 'Excellent'],
        'LightGBM': ['Gradient Boosting', 'Sequential', 'Optuna', 
                    'Focused', 'Medium', 'Fast', 'Medium', 'Good']
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Display comparison table
    st.markdown("### 📋 Model Characteristics Comparison")
    st.dataframe(df_comparison, use_container_width=True)
    
    # Performance metrics comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌳 Random Forest Advantages")
        st.markdown("""
        ✅ **Strengths:**
        - **Robust**: Less prone to overfitting
        - **Interpretable**: Clear feature importance
        - **Stable**: Consistent performance
        - **Parallel**: Can utilize all CPU cores
        - **Medical Grade**: 99.61% AUC-ROC
        
        ❌ **Weaknesses:**
        - Limited hyperparameter search space
        - Slower training time
        - Less sophisticated optimization
        """)
    
    with col2:
        st.markdown("### ⚡ LightGBM Advantages")
        st.markdown("""
        ✅ **Strengths:**
        - **Advanced Tuning**: Optuna Bayesian optimization
        - **Fast Training**: Early stopping capability
        - **Efficient**: Optimized for large datasets
        - **Flexible**: Comprehensive parameter space
        
        ❌ **Weaknesses:**
        - More complex to interpret
        - Higher overfitting risk
        - Parameter sensitive
        - Less balanced feature importance
        """)
    
    # Recommendation
    st.markdown("### 🏆 Recommendation")
    
    st.success("""
    **🎯 Production Recommendation: Use Random Forest**
    
    **Reasons:**
    1. **Superior Performance**: Better accuracy and precision
    2. **Medical Grade**: 99.61% AUC-ROC suitable for medical diagnosis
    3. **Interpretability**: Clear feature importance for medical professionals
    4. **Stability**: Consistent and reliable performance
    5. **Production Ready**: Robust implementation with comprehensive pipeline
    """)

def show_documentation_page():
    st.markdown("## 📋 Documentation")
    
    # GitHub repository link
    st.markdown("### 🔗 GitHub Repository")
    st.markdown("""
    **📂 Project Repository:** [Stroke Prediction Analysis](https://github.com/yourusername/stroke-prediction-rf)
    
    **🔧 Source Code:** All code, models, and documentation are available on GitHub
    **📊 Dataset:** Healthcare Dataset Stroke Data included in the repository
    **📝 Documentation:** Complete documentation and analysis files
    """)
    
    # Repository features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📁 Repository Structure:**
        - `app.py` - Streamlit web application
        - `src/main.py` - Main training pipeline
        - `models/` - Trained model files
        - `data/` - Dataset files
        - `*.md` - Documentation files
        """)
    
    with col2:
        st.markdown("""
        **🚀 Key Features:**
        - Complete source code
        - Pre-trained models
        - Comprehensive documentation
        - Ready for deployment
        - Open source project
        """)
    
    st.markdown("---")
    
    st.markdown("### 📚 Available Documentation Files")
    
    documentation_files = {
        'File': ['RINGKASAN_EKSEKUTIF.md', 'ANALISIS_KODE_PROGRAM.md', 
                'ANALISIS_TEKNIS_DETAIL.md', 'README.md'],
        'Purpose': ['Executive Summary', 'Comprehensive Analysis', 
                   'Technical Deep Dive', 'Main Documentation'],
        'Target Audience': ['Executives', 'Developers', 'Technical Leads', 'All Users'],
        'Content': ['Performance metrics, business impact', 
                   'Code structure, algorithm comparison',
                   'Code quality, best practices', 
                   'Usage instructions, features']
    }
    
    df_docs = pd.DataFrame(documentation_files)
    st.dataframe(df_docs, use_container_width=True)
    
    # Pipeline overview
    st.markdown("### 🔄 Pipeline Overview")
    
    pipeline_steps = [
        "1. Data Loading - Pemuatan dan eksplorasi data",
        "2. Target Analysis - Analisis distribusi target",
        "3. Missing Values - Group-based imputation untuk BMI",
        "4. Outlier Detection - IQR method untuk deteksi outlier",
        "5. Feature Engineering - Risk score, age groups, BMI categories",
        "6. Encoding - One-Hot Encoding untuk variabel kategorikal",
        "7. Feature Selection - SelectKBest (k=15) untuk seleksi fitur",
        "8. SMOTE Balancing - Penanganan class imbalance",
        "9. Data Splitting - Train/test split dengan stratifikasi",
        "10. Scaling - RobustScaler untuk normalisasi",
        "11. Hyperparameter Tuning - GridSearchCV (RF) vs Optuna (LightGBM)",
        "12. Model Training - Training dengan cross-validation",
        "13. Evaluation - Multiple metrics evaluation",
        "14. Model Saving - Persistence semua komponen"
    ]
    
    for step in pipeline_steps:
        st.markdown(f"• {step}")
    
    # Technical stack
    st.markdown("### 🛠️ Technical Stack")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Core Technologies:**
        - Python 3.8+
        - scikit-learn
        - LightGBM
        - Optuna
        - pandas, numpy
        - matplotlib, seaborn, plotly
        - imbalanced-learn
        - joblib
        """)
    
    with col2:
        st.markdown("""
        **Key Features:**
        - Advanced preprocessing pipeline
        - Hyperparameter optimization
        - Class imbalance handling
        - Comprehensive evaluation
        - Model persistence
        - Production-ready code
        """)
    
    # Contributing and license information
    st.markdown("### 🤝 Contributing")
    st.markdown("""
    **📝 How to Contribute:**
    1. Fork the repository on GitHub
    2. Create a feature branch (`git checkout -b feature/amazing-feature`)
    3. Commit your changes (`git commit -m 'Add amazing feature'`)
    4. Push to the branch (`git push origin feature/amazing-feature`)
    5. Open a Pull Request
    
    **🐛 Report Issues:** Create an issue on GitHub for bugs or feature requests
    **💡 Suggest Improvements:** Open discussions for new ideas
    **📚 Documentation:** Help improve documentation and examples
    """)
    
    # License information
    st.markdown("### 📄 License")
    st.markdown("""
    **MIT License** - This project is licensed under the MIT License.
    
    **📋 License Terms:**
    - ✅ Free to use for commercial and non-commercial purposes
    - ✅ Free to modify and distribute
    - ✅ Free to use for private and public projects
    - ✅ Attribution is appreciated but not required
    
    **⚠️ Disclaimer:** This model is for educational and research purposes. 
    For medical diagnosis, always consult healthcare professionals.
    """)
    
    # Contact information
    st.markdown("### 📞 Contact")
    st.markdown("""
    **📧 Questions or Support:**
    - Open an issue on GitHub
    - Create a discussion for general questions
    - Contact the maintainers through GitHub
    
    **🔗 Links:**
    - [GitHub Repository](https://github.com/yourusername/stroke-prediction-rf)
    - [Issues](https://github.com/yourusername/stroke-prediction-rf/issues)
    - [Discussions](https://github.com/yourusername/stroke-prediction-rf/discussions)
    """)

def show_prediction_page():
    st.markdown("## 🔮 Stroke Prediction Tool")
    st.markdown("### 🎯 Try Our Best Model (Random Forest)")
    
    # Load the best model
    try:
        model = joblib.load('models/random_forest_model_97.79%.pkl')
        scaler = joblib.load('models/scaler_97.79%.pkl')
        encoder = joblib.load('models/encoder_97.79%.pkl')
        feature_selector = joblib.load('models/feature_selector_97.79%.pkl')
        
        st.success("✅ Model loaded successfully!")
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Please ensure the model files are in the current directory")
        return
    
    # Input form
    st.markdown("### 📝 Enter Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👤 Demographics")
        age = st.slider("Age", 0, 100, 50)
        gender = st.selectbox("Gender", ["Male", "Female"])
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        
        st.markdown("#### 🏥 Medical History")
        hypertension = st.checkbox("Hypertension")
        heart_disease = st.checkbox("Heart Disease")
        ever_married = st.selectbox("Ever Married", ["Yes", "No"])
        
    with col2:
        st.markdown("#### 📊 Health Metrics")
        avg_glucose_level = st.slider("Average Glucose Level", 50.0, 300.0, 100.0, 0.1)
        
        # Weight and Height inputs
        st.markdown("#### ⚖️ Body Measurements")
        weight_kg = st.slider("Weight (kg)", 30.0, 200.0, 70.0, 0.1)
        height_cm = st.slider("Height (cm)", 100.0, 250.0, 170.0, 0.1)
        
        # Calculate BMI automatically
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        
        # Display calculated BMI
        st.markdown(f"**📊 Calculated BMI:** {bmi:.1f} kg/m²")
        
        smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])
        
        st.markdown("#### 📈 Additional Info")
        # Calculate risk factors (matching main.py categories)
        if age < 30:
            age_group = "Young"
        elif age < 45:
            age_group = "Adult"
        elif age < 60:
            age_group = "Middle-aged"
        elif age < 75:
            age_group = "Senior"
        else:
            age_group = "Elderly"
            
        if bmi < 18.5:
            bmi_category = "Underweight"
        elif bmi < 25:
            bmi_category = "Normal"
        elif bmi < 30:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obese"
            
        if avg_glucose_level < 100:
            glucose_category = "Normal"
        elif avg_glucose_level < 125:
            glucose_category = "Prediabetes"
        elif avg_glucose_level < 200:
            glucose_category = "Diabetes"
        else:
            glucose_category = "Very High"
        
        # Display calculated categories
        st.info(f"**Age Group:** {age_group}")
        st.info(f"**BMI Category:** {bmi_category}")
        st.info(f"**Glucose Category:** {glucose_category}")
        
        # Additional body measurements info
        st.info(f"**Weight:** {weight_kg:.1f} kg")
        st.info(f"**Height:** {height_cm:.1f} cm")
    
    # Enhanced risk score calculation (matching training pipeline)
    risk_score = 0
    # Age factors
    if age > 65: risk_score += 2
    if age > 75: risk_score += 1
    # Medical conditions
    if hypertension: risk_score += 2
    if heart_disease: risk_score += 2
    # Glucose levels
    if avg_glucose_level > 140: risk_score += 1
    if avg_glucose_level > 200: risk_score += 1
    # BMI factors
    if bmi > 30: risk_score += 1
    if bmi > 40: risk_score += 1
    # Smoking status
    if smoking_status in ["smokes", "formerly smoked"]: risk_score += 1
    
    st.markdown(f"### ⚠️ Calculated Risk Score: {risk_score}/12")
    
    # Prediction button
    if st.button("🔮 Predict Stroke Risk", type="primary"):
        with st.spinner("Analyzing patient data..."):
            try:
                # Prepare input data exactly as the model expects (without id column)
                input_data = {
                    'gender': [gender],
                    'age': [age],
                    'hypertension': [1 if hypertension else 0],
                    'heart_disease': [1 if heart_disease else 0],
                    'ever_married': [ever_married],
                    'work_type': [work_type],
                    'Residence_type': [residence_type],
                    'avg_glucose_level': [avg_glucose_level],
                    'bmi': [bmi],  # Calculated from weight and height
                    'smoking_status': [smoking_status]
                }
                
                # Create DataFrame
                df_input = pd.DataFrame(input_data)
                
                # Calculate enhanced risk score exactly as in training pipeline
                risk_factors = 0
                # Age factors (more granular)
                risk_factors += (df_input['age'] > 65).astype(int) * 2  # Elderly gets 2 points
                risk_factors += (df_input['age'] > 75).astype(int) * 1  # Very elderly gets extra point
                
                # Medical conditions
                risk_factors += df_input['hypertension'] * 2  # Hypertension gets 2 points
                risk_factors += df_input['heart_disease'] * 2  # Heart disease gets 2 points
                
                # Glucose levels (more granular)
                risk_factors += (df_input['avg_glucose_level'] > 140).astype(int) * 1  # High glucose
                risk_factors += (df_input['avg_glucose_level'] > 200).astype(int) * 1  # Very high glucose
                
                # BMI factors (more granular)
                risk_factors += (df_input['bmi'] > 30).astype(int) * 1  # Obese
                risk_factors += (df_input['bmi'] > 40).astype(int) * 1  # Severely obese
                
                # Smoking status
                risk_factors += (df_input['smoking_status'] == 'smokes').astype(int) * 1
                risk_factors += (df_input['smoking_status'] == 'formerly smoked').astype(int) * 1
                
                df_input['risk_score'] = risk_factors
                
                # Apply one-hot encoding to categorical variables
                categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
                encoded_data = encoder.transform(df_input[categorical_cols])
                encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
                
                # Combine with numerical features (without id column)
                numerical_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'risk_score']
                final_df = pd.concat([df_input[numerical_cols].reset_index(drop=True), 
                                    encoded_df.reset_index(drop=True)], axis=1)
                
                # Apply feature selection
                selected_features = feature_selector.transform(final_df)
                
                # Apply scaling
                scaled_features = scaler.transform(selected_features)
                
                # 6. Prediction
                prediction = model.predict(scaled_features)[0]
                prediction_proba = model.predict_proba(scaled_features)[0]
                
                # Display results
                st.markdown("## 🎯 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:
                        st.error("🚨 **HIGH RISK** - Stroke Likely")
                    else:
                        st.success("✅ **LOW RISK** - No Stroke")
                
                with col2:
                    stroke_prob = prediction_proba[1] * 100
                    st.metric("Stroke Probability", f"{stroke_prob:.2f}%")
                
                with col3:
                    no_stroke_prob = prediction_proba[0] * 100
                    st.metric("No Stroke Probability", f"{no_stroke_prob:.2f}%")
                
                # Risk assessment with adjusted thresholds
                st.markdown("### 📊 Risk Assessment")
                
                if stroke_prob >= 50:
                    risk_level = "🔴 **VERY HIGH RISK**"
                    recommendation = "Immediate medical consultation recommended"
                elif stroke_prob >= 35:
                    risk_level = "🟠 **HIGH RISK**"
                    recommendation = "Medical consultation advised within 1 week"
                elif stroke_prob >= 20:
                    risk_level = "🟡 **MODERATE RISK**"
                    recommendation = "Regular health monitoring recommended"
                elif stroke_prob >= 10:
                    risk_level = "🟢 **LOW RISK**"
                    recommendation = "Maintain healthy lifestyle"
                else:
                    risk_level = "🟢 **VERY LOW RISK**"
                    recommendation = "Continue healthy lifestyle"
                
                st.markdown(f"**Risk Level:** {risk_level}")
                st.markdown(f"**Recommendation:** {recommendation}")
                st.markdown(f"**Risk Score:** {risk_score}/5")
                
                # Feature importance for this prediction
                st.markdown("### 🔍 Key Factors Influencing This Prediction")
                
                # Get feature importance for this specific case
                feature_names = feature_selector.get_feature_names_out()
                importances = model.feature_importances_
                
                # Create feature importance DataFrame
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(10)
                
                # Plot feature importance
                fig = px.bar(importance_df, x='Importance', y='Feature', 
                           orientation='h', title="Top 10 Most Important Features")
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed analysis
                st.markdown("### 📋 Detailed Analysis")
                
                analysis_points = []
                if age > 65:
                    analysis_points.append("• **Age Factor**: Elderly age (>65) increases stroke risk")
                if hypertension:
                    analysis_points.append("• **Hypertension**: Major risk factor for stroke")
                if heart_disease:
                    analysis_points.append("• **Heart Disease**: Significant cardiovascular risk factor")
                if bmi > 30:
                    analysis_points.append("• **BMI**: Obesity increases stroke risk")
                if avg_glucose_level > 140:
                    analysis_points.append("• **Glucose**: High glucose levels indicate diabetes risk")
                if smoking_status in ["smokes", "formerly smoked"]:
                    analysis_points.append("• **Smoking**: Tobacco use increases stroke risk")
                
                # Add body measurements analysis
                st.markdown("#### 📏 Body Measurements Analysis")
                st.markdown(f"• **Weight**: {weight_kg:.1f} kg")
                st.markdown(f"• **Height**: {height_cm:.1f} cm")
                st.markdown(f"• **BMI**: {bmi:.1f} kg/m² ({bmi_category})")
                
                # BMI-specific recommendations
                if bmi > 30:
                    st.warning("⚠️ **Obesity Alert**: High BMI indicates increased stroke risk. Consider weight management.")
                elif bmi > 25:
                    st.info("ℹ️ **Overweight**: Moderate BMI. Regular exercise recommended.")
                elif bmi > 18.5:
                    st.success("✅ **Normal BMI**: Healthy weight range.")
                else:
                    st.warning("⚠️ **Underweight**: Low BMI may indicate health issues.")
                
                if analysis_points:
                    st.markdown("**Risk Factors Identified:**")
                    for point in analysis_points:
                        st.markdown(point)
                else:
                    st.success("✅ No major risk factors identified")
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.info("Please check your input data and try again")

if __name__ == "__main__":
    main() 