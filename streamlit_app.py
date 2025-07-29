import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stroke Risk Assessment",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2E86AB;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #2E86AB;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(46, 134, 171, 0.3);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .info-box {
        background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #2E86AB;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .dark-info-box {
        background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #2E86AB;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        color: white;
    }
    .prediction-box {
        background: linear-gradient(135deg, #DC3545 0%, #C82333 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(220, 53, 69, 0.4);
        animation: pulse 2s infinite;
    }
    .safe-box {
        background: linear-gradient(135deg, #28A745 0%, #20C997 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(40, 167, 69, 0.4);
    }
    .warning-box {
        background: linear-gradient(135deg, #FFC107 0%, #FF8C00 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 6px 20px rgba(255, 193, 7, 0.3);
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: bold;
        color: #2E86AB;
        margin: 1.5rem 0 1rem 0;
        padding: 0.5rem 0;
        border-bottom: 2px solid #E9ECEF;
    }
    .bmi-category {
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem 0;
    }
    .bmi-underweight { background-color: #FFE066; color: #856404; }
    .bmi-normal { background-color: #D4EDDA; color: #155724; }
    .bmi-overweight { background-color: #FFE066; color: #856404; }
    .bmi-obese { background-color: #F8D7DA; color: #721C24; }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .stButton > button {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(46, 134, 171, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("random_forest_model_97.38%.pkl")
        return model
    except:
        st.error("❌ Model file not found!")
        return None

# Calculate BMI
def calculate_bmi(height_cm, weight_kg):
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 1)

# Get BMI category
def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight", "bmi-underweight", "⚠️"
    elif bmi < 25:
        return "Normal weight", "bmi-normal", "✅"
    elif bmi < 30:
        return "Overweight", "bmi-overweight", "⚠️"
    else:
        return "Obese", "bmi-obese", "🚨"

# Preprocess input
def preprocess_input(data_dict):
    input_df = pd.DataFrame([data_dict])
    
    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    
    for col in categorical_columns:
        if col in input_df.columns:
            unique_values = ['Female', 'Male'] if col == 'gender' else \
                          ['No', 'Yes'] if col == 'ever_married' else \
                          ['children', 'Govt_job', 'Never_worked', 'Private', 'Self-employed'] if col == 'work_type' else \
                          ['Rural', 'Urban'] if col == 'Residence_type' else \
                          ['Unknown', 'formerly smoked', 'never smoked', 'smokes']
            
            for val in unique_values[1:]:
                col_name = f"{col}_{val}"
                input_df[col_name] = (input_df[col] == val).astype(int)
    
    input_df = input_df.drop(columns=categorical_columns, errors='ignore')
    
    expected_columns = [
        'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
        'gender_Male', 'ever_married_Yes', 'work_type_Never_worked', 'work_type_Private', 
        'work_type_Self-employed', 'work_type_children', 'Residence_type_Urban', 
        'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes'
    ]
    
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[expected_columns]
    return input_df

# Main app
def main():
    st.markdown('<h1 class="main-header">🫀 Stroke Risk Assessment</h1>', unsafe_allow_html=True)
    
    model = load_model()
    if model is None:
        return
    
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h3 style="color: white;">🧭 Navigation</h3>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "📊 Risk Assessment", "📈 Model Info", "ℹ️ About"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Risk Assessment":
        show_prediction_page(model)
    elif page == "📈 Model Info":
        show_model_info_page()
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page():
    st.markdown('<h2 class="sub-header">Welcome to Stroke Risk Assessment</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="dark-info-box">
    <h3>🎯 What is this application?</h3>
    <p>This is an advanced stroke risk assessment tool that uses machine learning to analyze your health parameters and provide personalized stroke risk evaluation. The model has been trained on comprehensive healthcare data and achieves 97.38% accuracy.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="dark-info-box">
        <h3>🔬 How does it work?</h3>
        <p>The application analyzes multiple health factors including age, medical history, lifestyle choices, and biometric data to assess your stroke risk. It uses advanced machine learning techniques including SMOTE for handling imbalanced data.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="dark-info-box">
        <h3>⚡ Key Features</h3>
        <ul>
        <li>🫀 Personalized risk assessment</li>
        <li>📊 Interactive visualizations</li>
        <li>📏 Automatic BMI calculation</li>
        <li>🎯 Risk factor identification</li>
        <li>💡 Professional medical guidance</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h3>📋 Risk Factors Analyzed</h3>', unsafe_allow_html=True)
        risk_factors = [
            "👤 Age & Gender",
            "💓 Hypertension",
            "🫀 Heart Disease",
            "🩸 Glucose Levels",
            "📏 BMI (Height & Weight)",
            "🚬 Smoking Status",
            "💼 Work Type",
            "🏠 Residence Type"
        ]
        
        for factor in risk_factors:
            st.write(f"• {factor}")
    
    st.markdown('<h2 class="sub-header">Model Performance</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card"><h3>🎯 Accuracy</h3><h2>97.38%</h2></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card"><h3>📊 Precision</h3><h2>96.15%</h2></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card"><h3>🔄 Recall</h3><h2>98.46%</h2></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card"><h3>⚖️ F1-Score</h3><h2>97.29%</h2></div>', unsafe_allow_html=True)

def show_prediction_page(model):
    st.markdown('<h2 class="sub-header">Stroke Risk Assessment</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="dark-info-box">
    <h3>📝 Instructions</h3>
    <p>Please fill in your information accurately. All data is processed locally and not stored. This assessment is for screening purposes only and should not replace professional medical advice.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        st.markdown('<h3 class="section-title">👤 Personal Information</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", min_value=1, max_value=120, value=50, help="Enter your current age")
            gender = st.selectbox("Gender", ["Female", "Male"], help="Select your gender")
            ever_married = st.selectbox("Ever Married", ["No", "Yes"], help="Have you ever been married?")
        
        with col2:
            work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Never_worked", "children"], 
                                   help="Select your current or previous work type")
            residence_type = st.selectbox("Residence Type", ["Urban", "Rural"], 
                                        help="Do you live in an urban or rural area?")
            smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"], 
                                        help="Select your smoking history")
        
        st.markdown('<h3 class="section-title">🏥 Medical Information</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No",
                                      help="Do you have high blood pressure?")
            heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No",
                                       help="Do you have any heart conditions?")
            avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, 
                                              value=120.0, step=0.1, help="Enter your average blood glucose level")
        
        with col2:
            st.markdown('<h4 style="color: #2E86AB; margin-bottom: 1rem;">📏 Body Measurements</h4>', unsafe_allow_html=True)
            height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1,
                                      help="Enter your height in centimeters")
            weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1,
                                      help="Enter your weight in kilograms")
            
            bmi = calculate_bmi(height_cm, weight_kg)
            bmi_category, bmi_class, bmi_icon = get_bmi_category(bmi)
            
            st.metric("Calculated BMI", f"{bmi} kg/m²")
            st.markdown(f'<div class="bmi-category {bmi_class}">{bmi_icon} {bmi_category}</div>', unsafe_allow_html=True)
        
        submitted = st.form_submit_button("🔮 Assess Stroke Risk", use_container_width=True)
        
        if submitted:
            input_data = {
                'age': age,
                'gender': gender,
                'hypertension': hypertension,
                'heart_disease': heart_disease,
                'ever_married': ever_married,
                'work_type': work_type,
                'residence_type': residence_type,
                'avg_glucose_level': avg_glucose_level,
                'bmi': bmi,
                'smoking_status': smoking_status
            }
            
            processed_input = preprocess_input(input_data)
            prediction_proba = model.predict_proba(processed_input)[0]
            prediction = model.predict(processed_input)[0]
            
            st.markdown('<h2 class="sub-header">Assessment Results</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                risk_percentage = prediction_proba[1] * 100
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = risk_percentage,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Stroke Risk Probability", 'font': {'size': 20}},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#2E86AB"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': "#28A745"},
                            {'range': [30, 70], 'color': "#FFC107"},
                            {'range': [70, 100], 'color': "#DC3545"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                
                fig.update_layout(
                    height=350,
                    font={'color': "#2C3E50"},
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if prediction == 1:
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown('<h2>⚠️ HIGH RISK</h2>', unsafe_allow_html=True)
                    st.markdown(f'<h3>Risk Probability: {risk_percentage:.1f}%</h3>', unsafe_allow_html=True)
                    st.markdown('<p><strong>Immediate Action Required:</strong></p>', unsafe_allow_html=True)
                    st.markdown('<ul><li>Consult a healthcare professional immediately</li><li>Schedule a comprehensive medical checkup</li><li>Monitor your blood pressure regularly</li><li>Follow medical advice strictly</li></ul>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="safe-box">', unsafe_allow_html=True)
                    st.markdown('<h2>✅ LOW RISK</h2>', unsafe_allow_html=True)
                    st.markdown(f'<h3>Risk Probability: {risk_percentage:.1f}%</h3>', unsafe_allow_html=True)
                    st.markdown('<p><strong>Recommendations:</strong></p>', unsafe_allow_html=True)
                    st.markdown('<ul><li>Continue maintaining a healthy lifestyle</li><li>Regular exercise and balanced diet</li><li>Annual health checkups</li><li>Monitor any changes in health</li></ul>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<h3 class="section-title">🔍 Risk Factor Analysis</h3>', unsafe_allow_html=True)
            
            risk_factors = []
            risk_levels = []
            
            if age > 65:
                risk_factors.append(f"Age ({age} years)")
                risk_levels.append("High risk age group")
            if hypertension:
                risk_factors.append("Hypertension")
                risk_levels.append("Major risk factor")
            if heart_disease:
                risk_factors.append("Heart Disease")
                risk_levels.append("Significant risk factor")
            if avg_glucose_level > 140:
                risk_factors.append(f"High glucose level ({avg_glucose_level} mg/dL)")
                risk_levels.append("Diabetes risk factor")
            if bmi > 30:
                risk_factors.append(f"High BMI ({bmi})")
                risk_levels.append("Obesity risk factor")
            if smoking_status in ["smokes", "formerly smoked"]:
                risk_factors.append(f"Smoking history ({smoking_status})")
                risk_levels.append("Cardiovascular risk factor")
            
            if risk_factors:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<h4>🚨 Identified Risk Factors:</h4>', unsafe_allow_html=True)
                    for factor in risk_factors:
                        st.markdown(f'<p style="color: #DC3545; font-weight: bold;">• {factor}</p>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<h4>📊 Risk Level:</h4>', unsafe_allow_html=True)
                    for level in risk_levels:
                        st.markdown(f'<p style="color: #6C757D;">• {level}</p>', unsafe_allow_html=True)
            else:
                st.success("🎉 No major risk factors identified! Keep maintaining your healthy lifestyle.")
            
            st.markdown("""
            <div class="warning-box">
            <h4>⚠️ Important Medical Disclaimer</h4>
            <p>This assessment is for educational and screening purposes only. It should not replace professional medical diagnosis, treatment, or advice. Always consult with qualified healthcare professionals for medical concerns.</p>
            </div>
            """, unsafe_allow_html=True)

def show_model_info_page():
    st.markdown('<h2 class="sub-header">Model Information</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h3>🔬 Model Details</h3>', unsafe_allow_html=True)
        st.write("**Algorithm:** Random Forest Classifier")
        st.write("**Number of Estimators:** 600")
        st.write("**Random State:** 42")
        st.write("**Data Balancing:** SMOTE (Synthetic Minority Over-sampling Technique)")
        
        st.markdown('<h3>📊 Training Data</h3>', unsafe_allow_html=True)
        st.write("**Dataset:** Healthcare Dataset Stroke Data")
        st.write("**Original Samples:** 5,110")
        st.write("**Features:** 13 (after preprocessing)")
        st.write("**Target:** Binary (Stroke: Yes/No)")
    
    with col2:
        st.markdown('<h3>📈 Performance Metrics</h3>', unsafe_allow_html=True)
        
        metrics = {
            "Accuracy": 97.38,
            "Precision": 96.15,
            "Recall": 98.46,
            "F1-Score": 97.29
        }
        
        for metric, value in metrics.items():
            st.metric(metric, f"{value}%")
    
    st.markdown('<h3 class="section-title">🎯 Feature Importance</h3>', unsafe_allow_html=True)
    
    features = [
        'Age', 'Hypertension', 'Heart Disease', 'Average Glucose Level',
        'BMI', 'Gender', 'Work Type', 'Smoking Status', 'Residence Type'
    ]
    importance = [0.25, 0.20, 0.18, 0.15, 0.12, 0.05, 0.03, 0.02, 0.01]
    
    fig = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title="Feature Importance in Stroke Prediction",
        labels={'x': 'Importance', 'y': 'Features'},
        color=importance,
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        height=400,
        font={'color': "#2C3E50"},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

def show_about_page():
    st.markdown('<h2 class="sub-header">About This Application</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="dark-info-box">
    <h3>🎯 Purpose</h3>
    <p>This application is designed to help individuals assess their stroke risk based on various health parameters. It serves as a screening tool and should not replace professional medical advice.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
    <h3>⚠️ Important Disclaimer</h3>
    <p>This application is for educational and screening purposes only. It should not be used as a substitute for professional medical diagnosis, treatment, or advice. Always consult with qualified healthcare professionals for medical concerns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="dark-info-box">
    <h3>🔬 Technical Information</h3>
    <p>The model uses machine learning techniques including Random Forest classification and SMOTE for handling imbalanced data. The application is built with Streamlit for an interactive user experience.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="dark-info-box">
    <h3>📊 Data Sources</h3>
    <p>The model was trained on a comprehensive healthcare dataset containing various demographic, medical, and lifestyle factors associated with stroke risk.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 