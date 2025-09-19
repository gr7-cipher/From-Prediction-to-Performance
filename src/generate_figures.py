"""
Figure Generation for Catalyst Discovery Paper
============================================

This module generates all the figures and plots required for the manuscript
and supplementary materials, including the R² vs training size plot,
modified correlation heatmap, and validation plots.

Authors: Research Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'sans-serif'
})

def generate_learning_curve_plot(model, X, y, title="Learning Curve", save_path=None):
    """
    Generate R² vs training set size plot for sensitivity analysis.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X : array-like
        Features
    y : array-like
        Targets
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Generated figure
    """
    # Define training sizes
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    # Calculate learning curve
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y, 
        train_sizes=train_sizes,
        cv=5,
        scoring='r2',
        random_state=42,
        n_jobs=-1
    )
    
    # Calculate means and standard deviations
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot training scores
    ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', 
            label='Training Score', linewidth=2, markersize=6)
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color='blue')
    
    # Plot validation scores
    ax.plot(train_sizes_abs, val_mean, 'o-', color='red', 
            label='Cross-Validation Score', linewidth=2, markersize=6)
    ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                    alpha=0.2, color='red')
    
    # Formatting
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Add text box with final performance
    final_val_score = val_mean[-1]
    final_val_std = val_std[-1]
    textstr = f'Final CV Score: {final_val_score:.3f} ± {final_val_std:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig

def generate_correlation_heatmap_with_pvalues(data, save_path=None):
    """
    Generate correlation heatmap with statistical significance (p-values).
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset with features and targets
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Generated figure
    """
    # Calculate correlation matrix
    correlation_matrix = data.corr()
    
    # Calculate p-values
    n_vars = len(data.columns)
    pvalue_matrix = pd.DataFrame(np.zeros((n_vars, n_vars)), 
                               columns=data.columns,
                               index=data.columns)
    
    for i, col1 in enumerate(data.columns):
        for j, col2 in enumerate(data.columns):
            if i != j:
                # Handle potential NaN values
                mask = ~(np.isnan(data[col1]) | np.isnan(data[col2]))
                if mask.sum() > 2:  # Need at least 3 points for correlation
                    corr, p_val = stats.pearsonr(data[col1][mask], data[col2][mask])
                    pvalue_matrix.iloc[i, j] = p_val
                else:
                    pvalue_matrix.iloc[i, j] = 1.0  # No significant correlation
            else:
                pvalue_matrix.iloc[i, j] = 0.0  # Perfect correlation with self
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Correlation heatmap
    mask_corr = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask_corr, annot=True, cmap='RdBu_r', 
                center=0, square=True, ax=ax1, 
                cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
                fmt='.2f', linewidths=0.5)
    ax1.set_title('Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    
    # P-value heatmap
    mask_pval = np.triu(np.ones_like(pvalue_matrix, dtype=bool))
    sns.heatmap(pvalue_matrix, mask=mask_pval, annot=True, cmap='viridis_r', 
                square=True, ax=ax2, 
                cbar_kws={'label': 'P-value', 'shrink': 0.8},
                fmt='.3f', linewidths=0.5)
    ax2.set_title('Statistical Significance (P-values)', fontsize=14, fontweight='bold', pad=20)
    
    # Add significance markers to p-value plot
    for i in range(len(pvalue_matrix.columns)):
        for j in range(len(pvalue_matrix.columns)):
            if not mask_pval[i, j]:  # Only for lower triangle
                p_val = pvalue_matrix.iloc[i, j]
                if p_val < 0.001:
                    ax2.text(j + 0.5, i + 0.5, '***', ha='center', va='center', 
                           color='white', fontweight='bold', fontsize=12)
                elif p_val < 0.01:
                    ax2.text(j + 0.5, i + 0.5, '**', ha='center', va='center', 
                           color='white', fontweight='bold', fontsize=12)
                elif p_val < 0.05:
                    ax2.text(j + 0.5, i + 0.5, '*', ha='center', va='center', 
                           color='white', fontweight='bold', fontsize=12)
    
    # Add legend for significance levels
    legend_text = "Significance levels:\n*** p < 0.001\n** p < 0.01\n* p < 0.05"
    ax2.text(1.02, 0.5, legend_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig

def generate_cross_validation_results_plot(cv_results_dict, save_path=None):
    """
    Generate cross-validation results visualization.
    
    Parameters:
    -----------
    cv_results_dict : dict
        Dictionary containing CV results for different models
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Generated figure
    """
    models = list(cv_results_dict.keys())
    metrics = ['r2', 'rmse', 'mae']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, metric in enumerate(metrics):
        means = [cv_results_dict[model][f'{metric}_mean'] for model in models]
        stds = [cv_results_dict[model][f'{metric}_std'] for model in models]
        
        x_pos = np.arange(len(models))
        bars = axes[i].bar(x_pos, means, yerr=stds, capsize=5, 
                          color=['skyblue', 'lightcoral'], alpha=0.8, edgecolor='black')
        
        axes[i].set_xlabel('Model', fontsize=12)
        axes[i].set_ylabel(metric.upper(), fontsize=12)
        axes[i].set_title(f'Cross-Validation {metric.upper()} Scores', fontsize=14, fontweight='bold')
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels(models, rotation=45)
        axes[i].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for j, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + std,
                        f'{mean:.3f}±{std:.3f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig

def generate_feature_importance_plot(importance_results, save_path=None):
    """
    Generate feature importance plot with statistical significance.
    
    Parameters:
    -----------
    importance_results : dict
        Feature importance results from permutation importance
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Generated figure
    """
    feature_names = importance_results['feature_names']
    importance_means = importance_results['importance_means']
    importance_stds = importance_results['importance_stds']
    p_values = importance_results['p_values']
    
    # Sort by importance
    sorted_indices = np.argsort(importance_means)[::-1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create colors based on significance
    colors = []
    for p_val in p_values:
        if p_val < 0.001:
            colors.append('darkred')
        elif p_val < 0.01:
            colors.append('red')
        elif p_val < 0.05:
            colors.append('orange')
        else:
            colors.append('lightgray')
    
    # Reorder based on sorted indices
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_means = [importance_means[i] for i in sorted_indices]
    sorted_stds = [importance_stds[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]
    sorted_pvals = [p_values[i] for i in sorted_indices]
    
    # Create horizontal bar plot
    y_pos = np.arange(len(sorted_names))
    bars = ax.barh(y_pos, sorted_means, xerr=sorted_stds, 
                   color=sorted_colors, alpha=0.8, edgecolor='black', capsize=3)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('Permutation Importance', fontsize=12)
    ax.set_title('Feature Importance with Statistical Significance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add significance markers
    for i, (bar, p_val) in enumerate(zip(bars, sorted_pvals)):
        width = bar.get_width()
        if p_val < 0.001:
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, '***',
                   ha='left', va='center', fontweight='bold', fontsize=12)
        elif p_val < 0.01:
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, '**',
                   ha='left', va='center', fontweight='bold', fontsize=12)
        elif p_val < 0.05:
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, '*',
                   ha='left', va='center', fontweight='bold', fontsize=12)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='darkred', alpha=0.8, label='p < 0.001 (***)'),
        plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.8, label='p < 0.01 (**)'),
        plt.Rectangle((0,0),1,1, facecolor='orange', alpha=0.8, label='p < 0.05 (*)'),
        plt.Rectangle((0,0),1,1, facecolor='lightgray', alpha=0.8, label='p ≥ 0.05 (n.s.)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig

def generate_kernel_comparison_plot(kernel_results, save_path=None):
    """
    Generate kernel comparison plot.
    
    Parameters:
    -----------
    kernel_results : dict
        Results from kernel comparison
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Generated figure
    """
    kernels = list(kernel_results.keys())
    metrics = ['log_marginal_likelihood', 'rmse', 'mae', 'r2']
    metric_labels = ['Log-Marginal Likelihood', 'RMSE (mV)', 'MAE (mV)', 'R²']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [kernel_results[kernel][metric] for kernel in kernels]
        
        bars = axes[i].bar(kernels, values, color=['skyblue', 'lightcoral'], 
                          alpha=0.8, edgecolor='black')
        
        axes[i].set_ylabel(label, fontsize=12)
        axes[i].set_title(f'Kernel Comparison: {label}', fontsize=14, fontweight='bold')
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}' if metric != 'log_marginal_likelihood' else f'{value:.1f}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Highlight better performance
        if metric in ['r2', 'log_marginal_likelihood']:
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('darkgoldenrod')
        bars[best_idx].set_linewidth(2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig

def generate_pareto_front_plot(pareto_solutions, save_path=None):
    """
    Generate Pareto front visualization.
    
    Parameters:
    -----------
    pareto_solutions : pd.DataFrame
        Pareto optimal solutions
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Generated figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all Pareto solutions
    scatter = ax.scatter(pareto_solutions['overpotential'], pareto_solutions['tafel_slope'],
                        c=pareto_solutions['composite_objective'], cmap='viridis_r',
                        s=100, alpha=0.8, edgecolors='black', linewidth=1)
    
    # Highlight best solution
    best_idx = pareto_solutions['composite_objective'].idxmin()
    best_solution = pareto_solutions.iloc[best_idx]
    
    ax.scatter(best_solution['overpotential'], best_solution['tafel_slope'],
              c='red', s=300, marker='*', edgecolors='darkred', linewidth=2,
              label=f'Best Solution\n({best_solution["overpotential"]:.1f} mV, {best_solution["tafel_slope"]:.1f} mV/dec)')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Composite Objective', fontsize=12)
    
    ax.set_xlabel('Overpotential (mV)', fontsize=12)
    ax.set_ylabel('Tafel Slope (mV/dec)', fontsize=12)
    ax.set_title('Pareto Front for Multi-Objective Catalyst Optimization', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig

def generate_sensitivity_analysis_plot(sensitivity_results, save_path=None):
    """
    Generate sensitivity analysis plot for different weight combinations.
    
    Parameters:
    -----------
    sensitivity_results : dict
        Results from sensitivity analysis
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Generated figure
    """
    weight_combinations = list(sensitivity_results.keys())
    overpotentials = [sensitivity_results[w]['best_overpotential'] for w in weight_combinations]
    tafel_slopes = [sensitivity_results[w]['best_tafel_slope'] for w in weight_combinations]
    dopants = [sensitivity_results[w]['best_dopant'] for w in weight_combinations]
    concentrations = [sensitivity_results[w]['best_concentration'] for w in weight_combinations]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Overpotential vs Weight Combination
    x_labels = [w.replace('_', '/') for w in weight_combinations]
    bars1 = ax1.bar(x_labels, overpotentials, color='skyblue', alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Best Overpotential (mV)', fontsize=12)
    ax1.set_xlabel('Weight Combination (η/TS)', fontsize=12)
    ax1.set_title('Sensitivity Analysis: Overpotential', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value, dopant, conc in zip(bars1, overpotentials, dopants, concentrations):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{value:.1f} mV\n{dopant} ({conc:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: Tafel Slope vs Weight Combination
    bars2 = ax2.bar(x_labels, tafel_slopes, color='lightcoral', alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Best Tafel Slope (mV/dec)', fontsize=12)
    ax2.set_xlabel('Weight Combination (η/TS)', fontsize=12)
    ax2.set_title('Sensitivity Analysis: Tafel Slope', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value, dopant, conc in zip(bars2, tafel_slopes, dopants, concentrations):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f} mV/dec\n{dopant} ({conc:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig

if __name__ == "__main__":
    # Example usage - this would normally be called from the main analysis script
    print("Figure generation module loaded successfully!")
    print("Available functions:")
    print("- generate_learning_curve_plot()")
    print("- generate_correlation_heatmap_with_pvalues()")
    print("- generate_cross_validation_results_plot()")
    print("- generate_feature_importance_plot()")
    print("- generate_kernel_comparison_plot()")
    print("- generate_pareto_front_plot()")
    print("- generate_sensitivity_analysis_plot()")

