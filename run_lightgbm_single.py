#!/usr/bin/env python3
"""
LightGBM Stroke Prediction - Single File Runner
===============================================

Simple script to run the complete LightGBM pipeline from a single file.

Usage:
    python run_lightgbm_single.py
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Run the LightGBM pipeline"""
    print("🚀 Starting LightGBM Stroke Prediction Pipeline...")
    
    try:
        # Import and run the LightGBM predictor
        from lightgbm_main_single import LightGBMStrokePredictor
        
        # Initialize predictor
        predictor = LightGBMStrokePredictor()
        
        # Run complete pipeline with 25 trials for faster execution
        results, model_filename = predictor.run_complete_pipeline(
            "data/healthcare-dataset-stroke-data.csv", 
            n_trials=25
        )
        
        print("\n" + "=" * 80)
        print("🎉 LIGHTGBM PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"📊 Final Results:")
        print(f"   Accuracy:  {results['accuracy'] * 100:.2f}%")
        print(f"   Precision: {results['precision'] * 100:.2f}%")
        print(f"   Recall:    {results['recall'] * 100:.2f}%")
        print(f"   F1-Score:  {results['f1'] * 100:.2f}%")
        print(f"   AUC-ROC:   {results['auc'] * 100:.2f}%")
        print(f"📁 Model saved as: {model_filename}")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error running LightGBM pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 