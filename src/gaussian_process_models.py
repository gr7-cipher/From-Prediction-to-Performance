"""
Gaussian Process Models for Catalyst Discovery
============================================

This module implements Gaussian Process regression models for predicting
overpotential and Tafel slope of metal-doped Co3O4 electrocatalysts.

Authors: Research Team
Date: 2025
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class CatalystGPModel:
    """
    Gaussian Process model for catalyst property prediction.
    
    This class implements the GP models described in the manuscript with
    composite kernels (Matern + WhiteNoise) and hyperparameter optimization.
    """
    
    def __init__(self, target_property='overpotential'):
        """
        Initialize the GP model.
        
        Parameters:
        -----------
        target_property : str
            Either 'overpotential' or 'tafel_slope'
        """
        self.target_property = target_property
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.is_fitted = False
        
        # Define composite kernel as described in Eq. 1
        self.kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * 
                      Matern(length_scale=1.0, nu=2.5) + 
                      WhiteKernel(noise_level=1e-5))
        
    def preprocess_data(self, X, y=None, fit_transform=True):
        """
        Preprocess the input data including one-hot encoding and scaling.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : array-like, optional
            Target values
        fit_transform : bool
            Whether to fit the transformers
            
        Returns:
        --------
        X_processed : np.ndarray
            Processed features
        y_processed : np.ndarray, optional
            Processed targets
        """
        X_processed = X.copy()
        
        # One-hot encode categorical variables (dopant material)
        categorical_cols = X_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if fit_transform:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col])
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    X_processed[col] = self.label_encoders[col].transform(X_processed[col])
        
        # Convert to numpy array
        X_processed = X_processed.values
        
        # Scale features
        if fit_transform:
            X_processed = self.scaler.fit_transform(X_processed)
        else:
            X_processed = self.scaler.transform(X_processed)
        
        if y is not None:
            y_processed = np.array(y)
            # Log-transform overpotential as mentioned in the manuscript
            if self.target_property == 'overpotential':
                y_processed = np.log(y_processed)
            return X_processed, y_processed
        
        return X_processed
    
    def fit(self, X, y):
        """
        Fit the Gaussian Process model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : array-like
            Target values
        """
        # Preprocess data
        X_processed, y_processed = self.preprocess_data(X, y, fit_transform=True)
        
        # Initialize GP model with optimized hyperparameters
        self.model = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-6,
            optimizer='fmin_l_bfgs_b',
            n_restarts_optimizer=10,
            normalize_y=True,
            random_state=42
        )
        
        # Fit the model
        self.model.fit(X_processed, y_processed)
        self.is_fitted = True
        
        print(f"Model fitted for {self.target_property}")
        print(f"Log-marginal likelihood: {self.model.log_marginal_likelihood():.2f}")
        print(f"Optimized kernel: {self.model.kernel_}")
        
    def predict(self, X, return_std=False):
        """
        Make predictions with the fitted model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        return_std : bool
            Whether to return prediction uncertainties
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted values
        std : np.ndarray, optional
            Prediction uncertainties
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_processed = self.preprocess_data(X, fit_transform=False)
        
        if return_std:
            pred_mean, pred_std = self.model.predict(X_processed, return_std=True)
            
            # Transform back from log space for overpotential
            if self.target_property == 'overpotential':
                pred_mean = np.exp(pred_mean)
                # Approximate std transformation (delta method)
                pred_std = pred_mean * pred_std
                
            return pred_mean, pred_std
        else:
            pred_mean = self.model.predict(X_processed)
            
            # Transform back from log space for overpotential
            if self.target_property == 'overpotential':
                pred_mean = np.exp(pred_mean)
                
            return pred_mean
    
    def cross_validate(self, X, y, cv_folds=10):
        """
        Perform k-fold cross-validation.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : array-like
            Target values
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        cv_results : dict
            Cross-validation metrics
        """
        X_processed, y_processed = self.preprocess_data(X, y, fit_transform=True)
        
        # Initialize model for CV
        cv_model = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-6,
            optimizer='fmin_l_bfgs_b',
            n_restarts_optimizer=10,
            normalize_y=True,
            random_state=42
        )
        
        # Perform cross-validation
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        r2_scores = []
        rmse_scores = []
        mae_scores = []
        
        for train_idx, val_idx in kf.split(X_processed):
            X_train, X_val = X_processed[train_idx], X_processed[val_idx]
            y_train, y_val = y_processed[train_idx], y_processed[val_idx]
            
            # Fit model
            cv_model.fit(X_train, y_train)
            
            # Predict
            y_pred = cv_model.predict(X_val)
            
            # Transform back if needed
            if self.target_property == 'overpotential':
                y_val_orig = np.exp(y_val)
                y_pred_orig = np.exp(y_pred)
            else:
                y_val_orig = y_val
                y_pred_orig = y_pred
            
            # Calculate metrics
            r2_scores.append(r2_score(y_val_orig, y_pred_orig))
            rmse_scores.append(np.sqrt(mean_squared_error(y_val_orig, y_pred_orig)))
            mae_scores.append(mean_absolute_error(y_val_orig, y_pred_orig))
        
        cv_results = {
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores),
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores),
            'r2_scores': r2_scores,
            'rmse_scores': rmse_scores,
            'mae_scores': mae_scores
        }
        
        return cv_results
    
    def calculate_feature_importance(self, X, y, n_permutations=50):
        """
        Calculate permutation feature importance.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : array-like
            Target values
        n_permutations : int
            Number of permutations for importance calculation
            
        Returns:
        --------
        importance_results : dict
            Feature importance results with statistical significance
        """
        if not self.is_fitted:
            self.fit(X, y)
        
        X_processed, y_processed = self.preprocess_data(X, y, fit_transform=False)
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            self.model, X_processed, y_processed,
            n_repeats=n_permutations,
            random_state=42,
            scoring='r2'
        )
        
        # Statistical significance testing
        feature_names = X.columns.tolist()
        importance_means = perm_importance.importances_mean
        importance_stds = perm_importance.importances_std
        
        # One-sample t-test against zero importance
        p_values = []
        for i in range(len(feature_names)):
            t_stat, p_val = stats.ttest_1samp(perm_importance.importances[i], 0)
            p_values.append(p_val)
        
        importance_results = {
            'feature_names': feature_names,
            'importance_means': importance_means,
            'importance_stds': importance_stds,
            'p_values': p_values,
            'significant_features': [name for name, p in zip(feature_names, p_values) if p < 0.05]
        }
        
        return importance_results
    
    def compare_kernels(self, X, y, kernels_to_compare=None):
        """
        Compare different kernel functions.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : array-like
            Target values
        kernels_to_compare : dict, optional
            Dictionary of kernel names and kernel objects
            
        Returns:
        --------
        comparison_results : dict
            Comparison results for different kernels
        """
        if kernels_to_compare is None:
            from sklearn.gaussian_process.kernels import RBF
            kernels_to_compare = {
                'Matern': ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5),
                'RBF': ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)
            }
        
        X_processed, y_processed = self.preprocess_data(X, y, fit_transform=True)
        
        results = {}
        
        for kernel_name, kernel in kernels_to_compare.items():
            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                optimizer='fmin_l_bfgs_b',
                n_restarts_optimizer=10,
                normalize_y=True,
                random_state=42
            )
            
            model.fit(X_processed, y_processed)
            
            # Calculate metrics
            y_pred = model.predict(X_processed)
            
            # Transform back if needed
            if self.target_property == 'overpotential':
                y_true_orig = np.exp(y_processed)
                y_pred_orig = np.exp(y_pred)
            else:
                y_true_orig = y_processed
                y_pred_orig = y_pred
            
            results[kernel_name] = {
                'log_marginal_likelihood': model.log_marginal_likelihood(),
                'rmse': np.sqrt(mean_squared_error(y_true_orig, y_pred_orig)),
                'mae': mean_absolute_error(y_true_orig, y_pred_orig),
                'r2': r2_score(y_true_orig, y_pred_orig)
            }
        
        return results


def create_sample_dataset():
    """
    Create a sample dataset for demonstration purposes.
    This represents the structure of the actual experimental data.
    """
    np.random.seed(42)
    
    # Define dopant materials
    dopants = ['Cu', 'Fe', 'Ni', 'Mn', 'Ce', 'Zn', 'Mo', 'V', 'Li', 'P', 'N', 'Te', 'Ru', 'Zr']
    
    # Generate sample data
    n_samples = 95
    data = []
    
    for i in range(n_samples):
        dopant = np.random.choice(dopants)
        
        # Dopant properties (example values)
        dopant_properties = {
            'Cu': {'valence_electrons': 11, 'atomic_radius': 1.28, 'ionic_radius': 0.73, 'covalent_radius': 1.32},
            'Fe': {'valence_electrons': 8, 'atomic_radius': 1.26, 'ionic_radius': 0.61, 'covalent_radius': 1.32},
            'Ni': {'valence_electrons': 10, 'atomic_radius': 1.24, 'ionic_radius': 0.69, 'covalent_radius': 1.24},
            'Mn': {'valence_electrons': 7, 'atomic_radius': 1.27, 'ionic_radius': 0.67, 'covalent_radius': 1.39},
            'Ce': {'valence_electrons': 4, 'atomic_radius': 1.82, 'ionic_radius': 1.01, 'covalent_radius': 2.04},
            'Zn': {'valence_electrons': 12, 'atomic_radius': 1.34, 'ionic_radius': 0.74, 'covalent_radius': 1.22},
            'Mo': {'valence_electrons': 6, 'atomic_radius': 1.39, 'ionic_radius': 0.69, 'covalent_radius': 1.54},
            'V': {'valence_electrons': 5, 'atomic_radius': 1.34, 'ionic_radius': 0.64, 'covalent_radius': 1.53},
            'Li': {'valence_electrons': 1, 'atomic_radius': 1.52, 'ionic_radius': 0.76, 'covalent_radius': 1.28},
            'P': {'valence_electrons': 5, 'atomic_radius': 1.10, 'ionic_radius': 0.44, 'covalent_radius': 1.07},
            'N': {'valence_electrons': 5, 'atomic_radius': 0.65, 'ionic_radius': 1.46, 'covalent_radius': 0.71},
            'Te': {'valence_electrons': 6, 'atomic_radius': 1.40, 'ionic_radius': 2.21, 'covalent_radius': 1.38},
            'Ru': {'valence_electrons': 8, 'atomic_radius': 1.34, 'ionic_radius': 0.68, 'covalent_radius': 1.46},
            'Zr': {'valence_electrons': 4, 'atomic_radius': 1.60, 'ionic_radius': 0.72, 'covalent_radius': 1.75}
        }
        
        sample = {
            'dopant_material': dopant,
            'doping_concentration': np.random.uniform(1, 55),
            'annealing_temperature': np.random.uniform(200, 550),
            'annealing_time': np.random.uniform(1, 4),
            'scan_rate': np.random.uniform(1, 50),
            'valence_electrons': dopant_properties[dopant]['valence_electrons'],
            'atomic_radius': dopant_properties[dopant]['atomic_radius'],
            'ionic_radius': dopant_properties[dopant]['ionic_radius'],
            'covalent_radius': dopant_properties[dopant]['covalent_radius']
        }
        
        # Generate synthetic overpotential and Tafel slope based on realistic relationships
        # These are simplified models for demonstration
        overpotential = (300 + 
                        np.random.normal(0, 50) + 
                        sample['doping_concentration'] * -2 +
                        (sample['atomic_radius'] - 1.3) * 100 +
                        (sample['annealing_temperature'] - 350) * 0.1)
        overpotential = max(200, min(500, overpotential))  # Realistic bounds
        
        tafel_slope = (90 + 
                      np.random.normal(0, 15) +
                      sample['doping_concentration'] * -0.3 +
                      (sample['ionic_radius'] - 0.7) * 20 +
                      (sample['annealing_time'] - 2.5) * 5)
        tafel_slope = max(60, min(150, tafel_slope))  # Realistic bounds
        
        sample['overpotential'] = overpotential
        sample['tafel_slope'] = tafel_slope
        
        data.append(sample)
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage
    print("Creating sample dataset...")
    df = create_sample_dataset()
    
    # Separate features and targets
    feature_cols = ['dopant_material', 'doping_concentration', 'annealing_temperature', 
                   'annealing_time', 'scan_rate', 'valence_electrons', 
                   'atomic_radius', 'ionic_radius', 'covalent_radius']
    
    X = df[feature_cols]
    y_overpotential = df['overpotential']
    y_tafel = df['tafel_slope']
    
    # Train overpotential model
    print("\nTraining overpotential model...")
    overpotential_model = CatalystGPModel('overpotential')
    overpotential_model.fit(X, y_overpotential)
    
    # Train Tafel slope model
    print("\nTraining Tafel slope model...")
    tafel_model = CatalystGPModel('tafel_slope')
    tafel_model.fit(X, y_tafel)
    
    # Perform cross-validation
    print("\nPerforming cross-validation...")
    cv_results_eta = overpotential_model.cross_validate(X, y_overpotential)
    cv_results_ts = tafel_model.cross_validate(X, y_tafel)
    
    print(f"Overpotential CV Results: R² = {cv_results_eta['r2_mean']:.3f} ± {cv_results_eta['r2_std']:.3f}")
    print(f"Tafel Slope CV Results: R² = {cv_results_ts['r2_mean']:.3f} ± {cv_results_ts['r2_std']:.3f}")
    
    # Calculate feature importance
    print("\nCalculating feature importance...")
    importance_eta = overpotential_model.calculate_feature_importance(X, y_overpotential)
    importance_ts = tafel_model.calculate_feature_importance(X, y_tafel)
    
    print("Significant features for overpotential:", importance_eta['significant_features'])
    print("Significant features for Tafel slope:", importance_ts['significant_features'])
    
    # Compare kernels
    print("\nComparing kernels...")
    kernel_comparison = overpotential_model.compare_kernels(X, y_overpotential)
    for kernel_name, metrics in kernel_comparison.items():
        print(f"{kernel_name}: LML = {metrics['log_marginal_likelihood']:.2f}, RMSE = {metrics['rmse']:.2f}")

