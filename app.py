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
        ["🏠 Home", "📈 Model Performance", "🔍 Feature Analysis", "🤖 Model Comparison", "📋 Documentation"]
    )
    
    if page == "🏠 Home":
        show_home_page()
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
            <p><strong>Accuracy:</strong> 97.74%</p>
            <p><strong>AUC-ROC:</strong> 99.61%</p>
            <p><strong>Precision:</strong> 99.57%</p>
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

if __name__ == "__main__":
    main() 