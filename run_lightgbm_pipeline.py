#!/usr/bin/env python3
"""
LightGBM Stroke Prediction Pipeline
===================================

This script runs the complete LightGBM pipeline including:
1. Data preprocessing
2. Hyperparameter optimization with Optuna
3. Model training
4. Evaluation and visualization

Usage:
    python run_lightgbm_pipeline.py
"""

import os
import sys
import time
import subprocess
import argparse

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 
        'sklearn', 'imblearn', 'joblib', 
        'streamlit', 'plotly', 'lightgbm', 'optuna'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def run_preprocessing():
    """Run the preprocessing pipeline"""
    print("\n" + "=" * 80)
    print("🔧 STEP 1: RUNNING PREPROCESSING PIPELINE")
    print("=" * 80)
    
    try:
        # Import and run preprocessing
        from src.lightgbm_preprocessing import LightGBMPreprocessor
        
        preprocessor = LightGBMPreprocessor()
        preprocessed_data = preprocessor.run_complete_pipeline("data/healthcare-dataset-stroke-data.csv")
        
        print("✅ Preprocessing completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {str(e)}")
        return False

def run_training():
    """Run the LightGBM training pipeline"""
    print("\n" + "=" * 80)
    print("🤖 STEP 2: RUNNING LIGHTGBM TRAINING PIPELINE")
    print("=" * 80)
    
    try:
        # Import and run training
        from src.lightgbm_training import LightGBMTrainer
        
        trainer = LightGBMTrainer()
        results = trainer.run_complete_training(n_trials=50)  # Reduced for faster execution
        
        if results:
            print("✅ Training completed successfully!")
            return True
        else:
            print("❌ Training failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        return False

def run_streamlit_app():
    """Run the Streamlit app"""
    print("\n" + "=" * 80)
    print("🌐 STEP 3: STARTING STREAMLIT APP")
    print("=" * 80)
    
    try:
        print("🚀 Starting Streamlit app...")
        print("📱 The app will open in your browser at: http://localhost:8501")
        print("🔄 To stop the app, press Ctrl+C")
        
        # Run streamlit app
        subprocess.run([
            "streamlit", "run", "streamlit_app_lightgbm.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Streamlit app stopped by user")
    except Exception as e:
        print(f"❌ Error running Streamlit app: {str(e)}")

def main():
    """Main function to run the complete pipeline"""
    parser = argparse.ArgumentParser(description='LightGBM Stroke Prediction Pipeline')
    parser.add_argument('--step', choices=['preprocessing', 'training', 'app', 'all'], 
                       default='all', help='Which step to run')
    parser.add_argument('--trials', type=int, default=50, 
                       help='Number of Optuna trials for hyperparameter optimization')
    parser.add_argument('--skip-deps', action='store_true', 
                       help='Skip dependency check')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 LIGHTGBM STROKE PREDICTION PIPELINE")
    print("=" * 80)
    
    start_time = time.time()
    
    # Check dependencies
    if not args.skip_deps:
        if not check_dependencies():
            print("❌ Please install missing dependencies and try again.")
            sys.exit(1)
    
    # Run selected steps
    if args.step in ['preprocessing', 'all']:
        if not run_preprocessing():
            print("❌ Preprocessing failed. Exiting.")
            sys.exit(1)
    
    if args.step in ['training', 'all']:
        if not run_training():
            print("❌ Training failed. Exiting.")
            sys.exit(1)
    
    if args.step in ['app', 'all']:
        run_streamlit_app()
    
    end_time = time.time()
    
    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"⏱️  Total execution time: {end_time - start_time:.2f} seconds")
    
    if args.step == 'all':
        print("\n📁 Generated files:")
        print("   - preprocessed_data.pkl (Preprocessed data)")
        print("   - preprocessors.pkl (Preprocessors)")
        print("   - lightgbm_model_*.pkl (Trained model)")
        print("   - lightgbm_results_*.pkl (Training results)")
        print("   - lightgbm_importance_*.pkl (Feature importance)")
        print("   - lightgbm_params_*.pkl (Best parameters)")
        print("   - lightgbm_analysis.html (Visualizations)")
        print("\n🌐 To run the web app:")
        print("   streamlit run streamlit_app_lightgbm.py")
    
    print("=" * 80)

if __name__ == "__main__":
    main() 