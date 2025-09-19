"""
Main Analysis Script for Catalyst Discovery
==========================================

This script runs the complete analysis workflow as described in the manuscript,
including data preprocessing, model training, validation, optimization, and
figure generation.

Authors: Research Team
Date: 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

# Import custom modules
from data_preprocessing import CatalystDataProcessor
from gaussian_process_models import CatalystGPModel, create_sample_dataset
from nsga_ii_optimization import CatalystOptimizer, create_optimization_bounds, run_sensitivity_analysis
from generate_figures import *

def main():
    """
    Main analysis workflow.
    """
    print("="*60)
    print("CATALYST DISCOVERY ANALYSIS WORKFLOW")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create results directory
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Step 1: Data Loading and Preprocessing
    print("\n" + "="*50)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*50)
    
    processor = CatalystDataProcessor()
    
    # Load data (using sample data for demonstration)
    print("Loading experimental data...")
    df_raw = processor.load_literature_data()
    print(f"Loaded {len(df_raw)} samples from literature")
    
    # Analyze raw data distribution
    print("Analyzing data distribution...")
    processor.analyze_data_distribution(df_raw, save_path=f"{results_dir}/data_distribution.png")
    
    # Preprocess data
    print("Preprocessing data...")
    X, y, preprocessing_report = processor.preprocess_data(df_raw)
    
    print("Preprocessing Summary:")
    for key, value in preprocessing_report.items():
        print(f"  {key}: {value}")
    
    # Generate correlation matrix with p-values (Supplementary Figure S1)
    print("Generating correlation matrix with p-values...")
    combined_data = pd.concat([X, y], axis=1)
    fig_corr = generate_correlation_heatmap_with_pvalues(
        combined_data, 
        save_path=f"{results_dir}/supplementary_figure_s1_correlation_pvalues.png"
    )
    
    # Step 2: Model Training and Validation
    print("\n" + "="*50)
    print("STEP 2: MODEL TRAINING AND VALIDATION")
    print("="*50)
    
    # Prepare feature columns
    feature_cols = ['dopant_material', 'doping_concentration', 'annealing_temperature', 
                   'annealing_time', 'scan_rate', 'valence_electrons', 
                   'atomic_radius', 'ionic_radius', 'covalent_radius']
    
    X_features = X[feature_cols]
    y_overpotential = y['overpotential']  # Already log-transformed
    y_tafel = y['tafel_slope']
    
    # Train overpotential model
    print("Training overpotential GP model...")
    overpotential_model = CatalystGPModel('overpotential')
    overpotential_model.fit(X_features, np.exp(y_overpotential))  # Convert back from log
    
    # Train Tafel slope model
    print("Training Tafel slope GP model...")
    tafel_model = CatalystGPModel('tafel_slope')
    tafel_model.fit(X_features, y_tafel)
    
    # Step 3: Cross-Validation and Model Validation
    print("\n" + "="*50)
    print("STEP 3: CROSS-VALIDATION AND MODEL VALIDATION")
    print("="*50)
    
    # Perform cross-validation
    print("Performing 10-fold cross-validation...")
    cv_results_eta = overpotential_model.cross_validate(X_features, np.exp(y_overpotential))
    cv_results_ts = tafel_model.cross_validate(X_features, y_tafel)
    
    print("Cross-Validation Results:")
    print(f"Overpotential Model:")
    print(f"  R² = {cv_results_eta['r2_mean']:.3f} ± {cv_results_eta['r2_std']:.3f}")
    print(f"  RMSE = {cv_results_eta['rmse_mean']:.3f} ± {cv_results_eta['rmse_std']:.3f} V")
    print(f"  MAE = {cv_results_eta['mae_mean']:.3f} ± {cv_results_eta['mae_std']:.3f} V")
    
    print(f"Tafel Slope Model:")
    print(f"  R² = {cv_results_ts['r2_mean']:.3f} ± {cv_results_ts['r2_std']:.3f}")
    print(f"  RMSE = {cv_results_ts['rmse_mean']:.3f} ± {cv_results_ts['rmse_std']:.3f} mV/dec")
    print(f"  MAE = {cv_results_ts['mae_mean']:.3f} ± {cv_results_ts['mae_std']:.3f} mV/dec")
    
    # Generate cross-validation results plot
    cv_results_dict = {
        'Overpotential': cv_results_eta,
        'Tafel Slope': cv_results_ts
    }
    fig_cv = generate_cross_validation_results_plot(
        cv_results_dict,
        save_path=f"{results_dir}/cross_validation_results.png"
    )
    
    # Generate learning curves (Figure S5)
    print("Generating learning curves for sensitivity analysis...")
    fig_lc_eta = generate_learning_curve_plot(
        overpotential_model.model, 
        overpotential_model.preprocess_data(X_features, fit_transform=False),
        overpotential_model.preprocess_data(X_features, np.exp(y_overpotential), fit_transform=False)[1],
        title="Learning Curve - Overpotential Model",
        save_path=f"{results_dir}/supplementary_figure_s5_learning_curve_overpotential.png"
    )
    
    fig_lc_ts = generate_learning_curve_plot(
        tafel_model.model,
        tafel_model.preprocess_data(X_features, fit_transform=False),
        tafel_model.preprocess_data(X_features, y_tafel, fit_transform=False)[1],
        title="Learning Curve - Tafel Slope Model",
        save_path=f"{results_dir}/supplementary_figure_s5_learning_curve_tafel.png"
    )
    
    # Step 4: Feature Importance Analysis
    print("\n" + "="*50)
    print("STEP 4: FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    # Calculate feature importance
    print("Calculating permutation importance...")
    importance_eta = overpotential_model.calculate_feature_importance(X_features, np.exp(y_overpotential))
    importance_ts = tafel_model.calculate_feature_importance(X_features, y_tafel)
    
    print("Significant features for overpotential:", importance_eta['significant_features'])
    print("Significant features for Tafel slope:", importance_ts['significant_features'])
    
    # Generate feature importance plots
    fig_imp_eta = generate_feature_importance_plot(
        importance_eta,
        save_path=f"{results_dir}/feature_importance_overpotential.png"
    )
    
    fig_imp_ts = generate_feature_importance_plot(
        importance_ts,
        save_path=f"{results_dir}/feature_importance_tafel_slope.png"
    )
    
    # Step 5: Kernel Comparison
    print("\n" + "="*50)
    print("STEP 5: KERNEL COMPARISON")
    print("="*50)
    
    # Compare kernels
    print("Comparing Matern vs RBF kernels...")
    kernel_comparison_eta = overpotential_model.compare_kernels(X_features, np.exp(y_overpotential))
    kernel_comparison_ts = tafel_model.compare_kernels(X_features, y_tafel)
    
    print("Kernel Comparison Results:")
    print("Overpotential Model:")
    for kernel_name, metrics in kernel_comparison_eta.items():
        print(f"  {kernel_name}: LML = {metrics['log_marginal_likelihood']:.2f}, "
              f"RMSE = {metrics['rmse']:.3f}, R² = {metrics['r2']:.3f}")
    
    print("Tafel Slope Model:")
    for kernel_name, metrics in kernel_comparison_ts.items():
        print(f"  {kernel_name}: LML = {metrics['log_marginal_likelihood']:.2f}, "
              f"RMSE = {metrics['rmse']:.3f}, R² = {metrics['r2']:.3f}")
    
    # Generate kernel comparison plots
    fig_kernel_eta = generate_kernel_comparison_plot(
        kernel_comparison_eta,
        save_path=f"{results_dir}/kernel_comparison_overpotential.png"
    )
    
    fig_kernel_ts = generate_kernel_comparison_plot(
        kernel_comparison_ts,
        save_path=f"{results_dir}/kernel_comparison_tafel_slope.png"
    )
    
    # Step 6: Multi-Objective Optimization
    print("\n" + "="*50)
    print("STEP 6: MULTI-OBJECTIVE OPTIMIZATION")
    print("="*50)
    
    # Setup optimization
    print("Setting up NSGA-II optimization...")
    optimizer = CatalystOptimizer(overpotential_model, tafel_model)
    feature_bounds, categorical_mappings = create_optimization_bounds()
    optimizer.setup_optimization(feature_bounds, categorical_mappings)
    
    # Run optimization
    print("Running optimization (50 generations, population size 50)...")
    result = optimizer.optimize(n_generations=50, verbose=True)
    
    # Get results
    print("Extracting optimization results...")
    pareto_solutions = optimizer.get_pareto_front()
    best_solution = optimizer.get_best_solution()
    
    print(f"\nOptimization Results:")
    print(f"Number of Pareto solutions: {len(pareto_solutions)}")
    print(f"Best solution:")
    print(f"  Dopant: {best_solution['dopant_material']}")
    print(f"  Concentration: {best_solution['doping_concentration']:.1f} wt.%")
    print(f"  Annealing Temperature: {best_solution['annealing_temperature']:.1f} °C")
    print(f"  Annealing Time: {best_solution['annealing_time']:.1f} hours")
    print(f"  Overpotential: {best_solution['overpotential']:.1f} mV")
    print(f"  Tafel Slope: {best_solution['tafel_slope']:.1f} mV/dec")
    print(f"  Composite Objective: {best_solution['composite_objective']:.3f}")
    
    # Generate Pareto front plot
    fig_pareto = generate_pareto_front_plot(
        pareto_solutions,
        save_path=f"{results_dir}/pareto_front.png"
    )
    
    # Plot convergence
    optimizer.plot_convergence(save_path=f"{results_dir}/optimization_convergence.png")
    
    # Step 7: Sensitivity Analysis
    print("\n" + "="*50)
    print("STEP 7: SENSITIVITY ANALYSIS")
    print("="*50)
    
    # Run sensitivity analysis for different weight combinations
    print("Running sensitivity analysis for different weight combinations...")
    weight_combinations = [(0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]
    sensitivity_results = run_sensitivity_analysis(optimizer, weight_combinations)
    
    print("Sensitivity Analysis Results:")
    for weights, results in sensitivity_results.items():
        eta_w, ts_w = weights.split('_')
        print(f"  Weights η={eta_w}, TS={ts_w}:")
        print(f"    Best dopant: {results['best_dopant']}")
        print(f"    Concentration: {results['best_concentration']:.1f} wt.%")
        print(f"    Overpotential: {results['best_overpotential']:.1f} mV")
        print(f"    Tafel Slope: {results['best_tafel_slope']:.1f} mV/dec")
    
    # Generate sensitivity analysis plot
    fig_sensitivity = generate_sensitivity_analysis_plot(
        sensitivity_results,
        save_path=f"{results_dir}/sensitivity_analysis.png"
    )
    
    # Step 8: Save Results and Models
    print("\n" + "="*50)
    print("STEP 8: SAVING RESULTS AND MODELS")
    print("="*50)
    
    # Save models
    print("Saving trained models...")
    with open(f"{results_dir}/overpotential_model.pkl", 'wb') as f:
        pickle.dump(overpotential_model, f)
    
    with open(f"{results_dir}/tafel_model.pkl", 'wb') as f:
        pickle.dump(tafel_model, f)
    
    # Save datasets
    print("Saving datasets...")
    df_raw.to_csv(f"{results_dir}/raw_dataset.csv", index=False)
    X_features.to_csv(f"{results_dir}/processed_features.csv", index=False)
    pd.DataFrame({'overpotential': np.exp(y_overpotential), 'tafel_slope': y_tafel}).to_csv(
        f"{results_dir}/processed_targets.csv", index=False)
    pareto_solutions.to_csv(f"{results_dir}/pareto_solutions.csv", index=False)
    
    # Save cross-validation results as Table S2
    cv_table = pd.DataFrame({
        'Model': ['Overpotential (η)', 'Tafel Slope (TS)'],
        'R²': [f"{cv_results_eta['r2_mean']:.3f} ± {cv_results_eta['r2_std']:.3f}",
               f"{cv_results_ts['r2_mean']:.3f} ± {cv_results_ts['r2_std']:.3f}"],
        'RMSE': [f"{cv_results_eta['rmse_mean']:.3f} ± {cv_results_eta['rmse_std']:.3f} V",
                 f"{cv_results_ts['rmse_mean']:.1f} ± {cv_results_ts['rmse_std']:.1f} mV/dec"],
        'MAE': [f"{cv_results_eta['mae_mean']:.3f} ± {cv_results_eta['mae_std']:.3f} V",
                f"{cv_results_ts['mae_mean']:.1f} ± {cv_results_ts['mae_std']:.1f} mV/dec"]
    })
    cv_table.to_csv(f"{results_dir}/supplementary_table_s2_cross_validation.csv", index=False)
    
    # Save kernel comparison results
    kernel_table = pd.DataFrame({
        'Kernel': ['Matern', 'RBF'],
        'Log-Marginal Likelihood (η)': [kernel_comparison_eta['Matern']['log_marginal_likelihood'],
                                        kernel_comparison_eta['RBF']['log_marginal_likelihood']],
        'RMSE (η) [mV]': [kernel_comparison_eta['Matern']['rmse'] * 1000,  # Convert to mV
                          kernel_comparison_eta['RBF']['rmse'] * 1000],
        'MAE (η) [mV]': [kernel_comparison_eta['Matern']['mae'] * 1000,
                         kernel_comparison_eta['RBF']['mae'] * 1000],
        'Log-Marginal Likelihood (TS)': [kernel_comparison_ts['Matern']['log_marginal_likelihood'],
                                         kernel_comparison_ts['RBF']['log_marginal_likelihood']],
        'RMSE (TS) [mV/dec]': [kernel_comparison_ts['Matern']['rmse'],
                               kernel_comparison_ts['RBF']['rmse']],
        'MAE (TS) [mV/dec]': [kernel_comparison_ts['Matern']['mae'],
                              kernel_comparison_ts['RBF']['mae']]
    })
    kernel_table.to_csv(f"{results_dir}/kernel_comparison_results.csv", index=False)
    
    # Generate summary report
    print("Generating summary report...")
    summary_report = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_info': {
            'total_samples': len(df_raw),
            'processed_samples': len(X_features),
            'features': feature_cols,
            'dopant_materials': list(df_raw['dopant_material'].unique())
        },
        'model_performance': {
            'overpotential': {
                'cv_r2': f"{cv_results_eta['r2_mean']:.3f} ± {cv_results_eta['r2_std']:.3f}",
                'cv_rmse': f"{cv_results_eta['rmse_mean']:.3f} ± {cv_results_eta['rmse_std']:.3f} V",
                'cv_mae': f"{cv_results_eta['mae_mean']:.3f} ± {cv_results_eta['mae_std']:.3f} V"
            },
            'tafel_slope': {
                'cv_r2': f"{cv_results_ts['r2_mean']:.3f} ± {cv_results_ts['r2_std']:.3f}",
                'cv_rmse': f"{cv_results_ts['rmse_mean']:.1f} ± {cv_results_ts['rmse_std']:.1f} mV/dec",
                'cv_mae': f"{cv_results_ts['mae_mean']:.1f} ± {cv_results_ts['mae_std']:.1f} mV/dec"
            }
        },
        'optimization_results': {
            'best_dopant': best_solution['dopant_material'],
            'best_concentration': f"{best_solution['doping_concentration']:.1f} wt.%",
            'best_overpotential': f"{best_solution['overpotential']:.1f} mV",
            'best_tafel_slope': f"{best_solution['tafel_slope']:.1f} mV/dec",
            'pareto_solutions_count': len(pareto_solutions)
        },
        'significant_features': {
            'overpotential': importance_eta['significant_features'],
            'tafel_slope': importance_ts['significant_features']
        }
    }
    
    # Save summary as JSON
    import json
    with open(f"{results_dir}/analysis_summary.json", 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Results saved in: {os.path.abspath(results_dir)}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # List generated files
    print("\nGenerated files:")
    for file in os.listdir(results_dir):
        print(f"  - {file}")
    
    return summary_report

if __name__ == "__main__":
    # Run the complete analysis
    summary = main()

