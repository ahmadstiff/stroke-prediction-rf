def load_data(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Fill missing BMI values by replacing them with the median BMI grouped by 'gender' and 'work_type'
    bmi = data.groupby(['gender', 'work_type'])['bmi'].transform(lambda x: x.fillna(x.median()))
    data['bmi'] = bmi
    
    # Drop rows where gender is 'Other'
    data = data[data['gender'] != 'Other']
    
    # Drop the 'id' column
    data.drop('id', axis=1, inplace=True)
    
    return data

def visualize_data(data):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Histograms for numerical features
    numerical_features = ['age', 'avg_glucose_level', 'bmi']
    data[numerical_features].hist(bins=20, figsize=(15, 10))
    plt.tight_layout()
    plt.show()
    
    # Countplots for categorical features
    categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married',
                            'work_type', 'Residence_type', 'smoking_status', 'stroke']
    plt.figure(figsize=(20, 15))
    for i, col in enumerate(categorical_features):
        plt.subplot(4, 2, i+1)
        sns.countplot(x=col, data=data)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()