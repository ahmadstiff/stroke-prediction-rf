#!/usr/bin/env python3
"""
Model Comparison: Random Forest vs LightGBM
===========================================

This script compares the performance of Random Forest and LightGBM models
for stroke prediction.

Usage:
    python compare_models.py
"""

import sys
import os
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_random_forest():
    """Run Random Forest pipeline"""
    print("🌲 Running Random Forest Pipeline...")
    
    try:
        from main import StrokeDataPreprocessor
        
        # Initialize preprocessor
        preprocessor = StrokeDataPreprocessor()
        
        # Run complete pipeline
        start_time = time.time()
        results, model_filename = preprocessor.run_complete_pipeline("data/healthcare-dataset-stroke-data.csv")
        end_time = time.time()
        
        rf_results = {
            'model': 'Random Forest',
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'auc': results['auc'],
            'execution_time': end_time - start_time,
            'model_filename': model_filename
        }
        
        print(f"✅ Random Forest completed in {rf_results['execution_time']:.2f} seconds")
        return rf_results
        
    except Exception as e:
        print(f"❌ Error running Random Forest: {str(e)}")
        return None

def run_lightgbm():
    """Run LightGBM pipeline"""
    print("💡 Running LightGBM Pipeline...")
    
    try:
        from lightgbm_main import LightGBMStrokePredictor
        
        # Initialize predictor
        predictor = LightGBMStrokePredictor()
        
        # Run complete pipeline
        start_time = time.time()
        results, model_filename = predictor.run_complete_pipeline("data/healthcare-dataset-stroke-data.csv", n_trials=50)
        end_time = time.time()
        
        lgb_results = {
            'model': 'LightGBM',
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'auc': results['auc'],
            'execution_time': end_time - start_time,
            'model_filename': model_filename
        }
        
        print(f"✅ LightGBM completed in {lgb_results['execution_time']:.2f} seconds")
        return lgb_results
        
    except Exception as e:
        print(f"❌ Error running LightGBM: {str(e)}")
        return None

def create_comparison_visualization(rf_results, lgb_results):
    """Create comparison visualization"""
    print("\n📈 Creating comparison visualization...")
    
    # Create comparison data
    models = ['Random Forest', 'LightGBM']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    rf_values = [
        rf_results['accuracy'] * 100,
        rf_results['precision'] * 100,
        rf_results['recall'] * 100,
        rf_results['f1'] * 100,
        rf_results['auc'] * 100
    ]
    
    lgb_values = [
        lgb_results['accuracy'] * 100,
        lgb_results['precision'] * 100,
        lgb_results['recall'] * 100,
        lgb_results['f1'] * 100,
        lgb_results['auc'] * 100
    ]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Performance Metrics Comparison', 'Execution Time', 'Accuracy Comparison', 'AUC-ROC Comparison'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # 1. Performance Metrics Comparison
    fig.add_trace(
        go.Bar(name='Random Forest', x=metrics, y=rf_values, marker_color='blue'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='LightGBM', x=metrics, y=lgb_values, marker_color='green'),
        row=1, col=1
    )
    
    # 2. Execution Time
    execution_times = [rf_results['execution_time'], lgb_results['execution_time']]
    fig.add_trace(
        go.Bar(name='Execution Time (seconds)', x=models, y=execution_times, marker_color='orange'),
        row=1, col=2
    )
    
    # 3. Accuracy Comparison
    accuracies = [rf_results['accuracy'] * 100, lgb_results['accuracy'] * 100]
    fig.add_trace(
        go.Bar(name='Accuracy (%)', x=models, y=accuracies, marker_color='purple'),
        row=2, col=1
    )
    
    # 4. AUC-ROC Comparison
    aucs = [rf_results['auc'] * 100, lgb_results['auc'] * 100]
    fig.add_trace(
        go.Bar(name='AUC-ROC (%)', x=models, y=aucs, marker_color='red'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Random Forest vs LightGBM Performance Comparison",
        height=800,
        showlegend=True
    )
    
    # Save plot
    fig.write_html("model_comparison.html")
    print("✅ Comparison visualization saved as: model_comparison.html")
    
    return fig

def print_comparison_table(rf_results, lgb_results):
    """Print comparison table"""
    print("\n" + "=" * 80)
    print("📊 MODEL COMPARISON RESULTS")
    print("=" * 80)
    
    # Create comparison table
    comparison_data = {
        'Metric': ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'AUC-ROC (%)', 'Execution Time (s)'],
        'Random Forest': [
            rf_results['accuracy'] * 100,
            rf_results['precision'] * 100,
            rf_results['recall'] * 100,
            rf_results['f1'] * 100,
            rf_results['auc'] * 100,
            rf_results['execution_time']
        ],
        'LightGBM': [
            lgb_results['accuracy'] * 100,
            lgb_results['precision'] * 100,
            lgb_results['recall'] * 100,
            lgb_results['f1'] * 100,
            lgb_results['auc'] * 100,
            lgb_results['execution_time']
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    # Determine winner for each metric
    print("\n🏆 WINNER ANALYSIS:")
    print("=" * 50)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    rf_wins = 0
    lgb_wins = 0
    
    for metric, name in zip(metrics, metric_names):
        rf_value = rf_results[metric] * 100
        lgb_value = lgb_results[metric] * 100
        
        if rf_value > lgb_value:
            winner = "Random Forest"
            rf_wins += 1
        elif lgb_value > rf_value:
            winner = "LightGBM"
            lgb_wins += 1
        else:
            winner = "Tie"
        
        print(f"{name:12}: {winner}")
    
    print("\n" + "=" * 50)
    print(f"Random Forest wins: {rf_wins}")
    print(f"LightGBM wins: {lgb_wins}")
    
    if rf_wins > lgb_wins:
        print("🏆 OVERALL WINNER: Random Forest")
    elif lgb_wins > rf_wins:
        print("🏆 OVERALL WINNER: LightGBM")
    else:
        print("🏆 OVERALL RESULT: Tie")
    
    print("=" * 50)

def main():
    """Main comparison function"""
    print("🚀 Starting Model Comparison: Random Forest vs LightGBM")
    print("=" * 80)
    
    # Run Random Forest
    rf_results = run_random_forest()
    if rf_results is None:
        print("❌ Random Forest failed. Exiting.")
        return 1
    
    print("\n" + "=" * 80)
    
    # Run LightGBM
    lgb_results = run_lightgbm()
    if lgb_results is None:
        print("❌ LightGBM failed. Exiting.")
        return 1
    
    print("\n" + "=" * 80)
    
    # Create comparison visualization
    create_comparison_visualization(rf_results, lgb_results)
    
    # Print comparison table
    print_comparison_table(rf_results, lgb_results)
    
    print("\n" + "=" * 80)
    print("🎉 MODEL COMPARISON COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("📁 Generated files:")
    print("   - model_comparison.html (Comparison visualization)")
    print("   - random_forest_model_*.pkl (Random Forest model)")
    print("   - lightgbm_model_*.pkl (LightGBM model)")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    exit(main()) 