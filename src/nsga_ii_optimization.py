"""
NSGA-II Multi-Objective Optimization for Catalyst Discovery
=========================================================

This module implements the NSGA-II genetic algorithm for multi-objective
optimization of catalyst parameters to minimize overpotential and Tafel slope.

Authors: Research Team
Date: 2025
"""

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt
from gaussian_process_models import CatalystGPModel
import pickle
import warnings
warnings.filterwarnings('ignore')

class CatalystOptimizationProblem(Problem):
    """
    Multi-objective optimization problem for catalyst design.
    
    This class defines the optimization problem as described in the manuscript,
    using GP surrogate models to predict overpotential and Tafel slope.
    """
    
    def __init__(self, overpotential_model, tafel_model, feature_bounds, categorical_mappings):
        """
        Initialize the optimization problem.
        
        Parameters:
        -----------
        overpotential_model : CatalystGPModel
            Trained GP model for overpotential prediction
        tafel_model : CatalystGPModel
            Trained GP model for Tafel slope prediction
        feature_bounds : dict
            Bounds for continuous variables
        categorical_mappings : dict
            Mappings for categorical variables
        """
        self.overpotential_model = overpotential_model
        self.tafel_model = tafel_model
        self.feature_bounds = feature_bounds
        self.categorical_mappings = categorical_mappings
        
        # Define problem dimensions
        n_var = len(feature_bounds)
        n_obj = 2  # overpotential and Tafel slope
        n_constr = 0
        
        # Extract bounds
        xl = [bounds[0] for bounds in feature_bounds.values()]
        xu = [bounds[1] for bounds in feature_bounds.values()]
        
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)
        
        # Store feature names for reconstruction
        self.feature_names = list(feature_bounds.keys())
        
        # Normalization parameters (from training data)
        self.eta_min = 200.0  # Minimum overpotential in dataset
        self.eta_max = 500.0  # Maximum overpotential in dataset
        self.ts_min = 60.0    # Minimum Tafel slope in dataset
        self.ts_max = 150.0   # Maximum Tafel slope in dataset
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate the objective functions for given parameter sets.
        
        Parameters:
        -----------
        X : np.ndarray
            Parameter sets to evaluate (n_samples x n_variables)
        out : dict
            Output dictionary to store results
        """
        n_samples = X.shape[0]
        objectives = np.zeros((n_samples, 2))
        
        for i in range(n_samples):
            # Convert optimization variables to feature DataFrame
            features = self._variables_to_features(X[i])
            
            # Predict overpotential and Tafel slope
            eta_pred = self.overpotential_model.predict(features)[0]
            ts_pred = self.tafel_model.predict(features)[0]
            
            # Normalize objectives as described in Eq. 2
            eta_norm = (eta_pred - self.eta_min) / (self.eta_max - self.eta_min)
            ts_norm = (ts_pred - self.ts_min) / (self.ts_max - self.ts_min)
            
            # Composite objective function as described in Eq. 3
            # Note: We minimize the composite objective, so we use it directly
            composite_objective = 0.70 * eta_norm + 0.30 * ts_norm
            
            # Store both individual objectives and composite
            objectives[i, 0] = eta_pred  # Overpotential (mV)
            objectives[i, 1] = ts_pred   # Tafel slope (mV/dec)
        
        out["F"] = objectives
    
    def _variables_to_features(self, variables):
        """
        Convert optimization variables to feature DataFrame.
        
        Parameters:
        -----------
        variables : np.ndarray
            Optimization variables
            
        Returns:
        --------
        features : pd.DataFrame
            Feature DataFrame for model prediction
        """
        feature_dict = {}
        
        for i, (feature_name, value) in enumerate(zip(self.feature_names, variables)):
            if feature_name == 'dopant_material':
                # Convert continuous variable back to categorical
                dopant_idx = int(np.round(value))
                dopant_idx = np.clip(dopant_idx, 0, len(self.categorical_mappings['dopant_material']) - 1)
                feature_dict[feature_name] = self.categorical_mappings['dopant_material'][dopant_idx]
            else:
                feature_dict[feature_name] = value
        
        return pd.DataFrame([feature_dict])


class CatalystOptimizer:
    """
    Main optimizer class that coordinates the multi-objective optimization.
    """
    
    def __init__(self, overpotential_model, tafel_model):
        """
        Initialize the optimizer.
        
        Parameters:
        -----------
        overpotential_model : CatalystGPModel
            Trained GP model for overpotential prediction
        tafel_model : CatalystGPModel
            Trained GP model for Tafel slope prediction
        """
        self.overpotential_model = overpotential_model
        self.tafel_model = tafel_model
        self.problem = None
        self.algorithm = None
        self.result = None
        
    def setup_optimization(self, feature_bounds, categorical_mappings):
        """
        Setup the optimization problem and algorithm.
        
        Parameters:
        -----------
        feature_bounds : dict
            Bounds for continuous variables
        categorical_mappings : dict
            Mappings for categorical variables
        """
        # Create optimization problem
        self.problem = CatalystOptimizationProblem(
            self.overpotential_model,
            self.tafel_model,
            feature_bounds,
            categorical_mappings
        )
        
        # Setup NSGA-II algorithm as described in the manuscript
        self.algorithm = NSGA2(
            pop_size=50,  # Population size as mentioned in manuscript
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(prob=1.0/len(feature_bounds), eta=20),
            eliminate_duplicates=True
        )
    
    def optimize(self, n_generations=50, verbose=True):
        """
        Run the optimization process.
        
        Parameters:
        -----------
        n_generations : int
            Number of generations (as mentioned in manuscript)
        verbose : bool
            Whether to print progress
            
        Returns:
        --------
        result : pymoo.core.result.Result
            Optimization result
        """
        if self.problem is None or self.algorithm is None:
            raise ValueError("Must call setup_optimization() first")
        
        # Run optimization
        self.result = minimize(
            self.problem,
            self.algorithm,
            ('n_gen', n_generations),
            verbose=verbose,
            seed=42
        )
        
        return self.result
    
    def get_pareto_front(self):
        """
        Extract the Pareto front from optimization results.
        
        Returns:
        --------
        pareto_solutions : pd.DataFrame
            Pareto optimal solutions with their objectives
        """
        if self.result is None:
            raise ValueError("Must run optimization first")
        
        # Extract Pareto optimal solutions
        pareto_X = self.result.X
        pareto_F = self.result.F
        
        # Convert to DataFrame
        solutions = []
        for i in range(len(pareto_X)):
            # Convert variables to features
            features = self.problem._variables_to_features(pareto_X[i])
            solution = features.iloc[0].to_dict()
            solution['overpotential'] = pareto_F[i, 0]
            solution['tafel_slope'] = pareto_F[i, 1]
            
            # Calculate composite objective
            eta_norm = (pareto_F[i, 0] - self.problem.eta_min) / (self.problem.eta_max - self.problem.eta_min)
            ts_norm = (pareto_F[i, 1] - self.problem.ts_min) / (self.problem.ts_max - self.problem.ts_min)
            solution['composite_objective'] = 0.70 * eta_norm + 0.30 * ts_norm
            
            solutions.append(solution)
        
        return pd.DataFrame(solutions)
    
    def get_best_solution(self):
        """
        Get the best solution based on the composite objective function.
        
        Returns:
        --------
        best_solution : dict
            Best solution parameters and objectives
        """
        pareto_front = self.get_pareto_front()
        best_idx = pareto_front['composite_objective'].idxmin()
        return pareto_front.iloc[best_idx].to_dict()
    
    def plot_pareto_front(self, save_path=None):
        """
        Plot the Pareto front.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if self.result is None:
            raise ValueError("Must run optimization first")
        
        plt.figure(figsize=(10, 8))
        
        # Plot Pareto front
        pareto_F = self.result.F
        plt.scatter(pareto_F[:, 0], pareto_F[:, 1], 
                   c='red', s=100, alpha=0.7, label='Pareto Front')
        
        # Highlight best solution
        best_solution = self.get_best_solution()
        plt.scatter(best_solution['overpotential'], best_solution['tafel_slope'],
                   c='gold', s=200, marker='*', edgecolors='black', linewidth=2,
                   label=f'Best Solution\n(η={best_solution["overpotential"]:.1f} mV, TS={best_solution["tafel_slope"]:.1f} mV/dec)')
        
        plt.xlabel('Overpotential (mV)', fontsize=12)
        plt.ylabel('Tafel Slope (mV/dec)', fontsize=12)
        plt.title('Pareto Front for Catalyst Optimization', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_convergence(self, save_path=None):
        """
        Plot the convergence of the optimization algorithm.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if self.result is None:
            raise ValueError("Must run optimization first")
        
        # Extract convergence data
        n_gen = len(self.result.history)
        best_objectives = []
        
        for gen in range(n_gen):
            gen_F = self.result.history[gen].result.F
            # Calculate composite objectives for this generation
            composite_objs = []
            for i in range(len(gen_F)):
                eta_norm = (gen_F[i, 0] - self.problem.eta_min) / (self.problem.eta_max - self.problem.eta_min)
                ts_norm = (gen_F[i, 1] - self.problem.ts_min) / (self.problem.ts_max - self.problem.ts_min)
                composite_obj = 0.70 * eta_norm + 0.30 * ts_norm
                composite_objs.append(composite_obj)
            
            best_objectives.append(min(composite_objs))
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, n_gen + 1), best_objectives, 'b-', linewidth=2, marker='o')
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Best Composite Objective', fontsize=12)
        plt.title('Optimization Convergence', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_optimization_bounds():
    """
    Create the bounds for optimization variables as described in the manuscript.
    
    Returns:
    --------
    feature_bounds : dict
        Bounds for each optimization variable
    categorical_mappings : dict
        Mappings for categorical variables
    """
    # Define bounds based on manuscript
    feature_bounds = {
        'dopant_material': (0, 13),  # Will be mapped to categorical
        'doping_concentration': (1, 55),  # wt.%
        'annealing_temperature': (200, 550),  # °C
        'annealing_time': (1, 4),  # hours
        'scan_rate': (1, 50),  # mV/s
        'valence_electrons': (1, 12),  # Will be determined by dopant
        'atomic_radius': (0.65, 1.82),  # Å
        'ionic_radius': (0.44, 2.21),  # Å
        'covalent_radius': (0.71, 2.04)  # Å
    }
    
    # Categorical mappings
    categorical_mappings = {
        'dopant_material': ['Ce', 'Cu', 'Fe', 'Li', 'Mn', 'Mo', 'N', 'Ni', 'P', 'Ru', 'Te', 'V', 'Zn', 'Zr']
    }
    
    return feature_bounds, categorical_mappings


def run_sensitivity_analysis(optimizer, weight_combinations):
    """
    Run sensitivity analysis for different weight combinations.
    
    Parameters:
    -----------
    optimizer : CatalystOptimizer
        Configured optimizer
    weight_combinations : list
        List of (eta_weight, ts_weight) tuples
        
    Returns:
    --------
    sensitivity_results : dict
        Results for different weight combinations
    """
    sensitivity_results = {}
    
    for eta_weight, ts_weight in weight_combinations:
        print(f"Running optimization with weights: η={eta_weight:.1f}, TS={ts_weight:.1f}")
        
        # Modify the problem's objective function weights
        original_problem = optimizer.problem
        
        # Create a modified problem class
        class ModifiedProblem(CatalystOptimizationProblem):
            def _evaluate(self, X, out, *args, **kwargs):
                n_samples = X.shape[0]
                objectives = np.zeros((n_samples, 2))
                
                for i in range(n_samples):
                    features = self._variables_to_features(X[i])
                    eta_pred = self.overpotential_model.predict(features)[0]
                    ts_pred = self.tafel_model.predict(features)[0]
                    
                    objectives[i, 0] = eta_pred
                    objectives[i, 1] = ts_pred
                
                out["F"] = objectives
        
        # Run optimization with modified weights
        modified_problem = ModifiedProblem(
            optimizer.overpotential_model,
            optimizer.tafel_model,
            optimizer.problem.feature_bounds,
            optimizer.problem.categorical_mappings
        )
        
        optimizer.problem = modified_problem
        result = optimizer.optimize(n_generations=30, verbose=False)
        
        # Store results
        best_solution = optimizer.get_best_solution()
        sensitivity_results[f"{eta_weight:.1f}_{ts_weight:.1f}"] = {
            'best_overpotential': best_solution['overpotential'],
            'best_tafel_slope': best_solution['tafel_slope'],
            'best_dopant': best_solution['dopant_material'],
            'best_concentration': best_solution['doping_concentration']
        }
        
        # Restore original problem
        optimizer.problem = original_problem
    
    return sensitivity_results


if __name__ == "__main__":
    # Example usage
    print("Loading trained models...")
    
    # This would normally load pre-trained models
    # For demonstration, we'll create and train them
    from gaussian_process_models import create_sample_dataset, CatalystGPModel
    
    # Create sample dataset
    df = create_sample_dataset()
    
    # Prepare features and targets
    feature_cols = ['dopant_material', 'doping_concentration', 'annealing_temperature', 
                   'annealing_time', 'scan_rate', 'valence_electrons', 
                   'atomic_radius', 'ionic_radius', 'covalent_radius']
    
    X = df[feature_cols]
    y_overpotential = df['overpotential']
    y_tafel = df['tafel_slope']
    
    # Train models
    print("Training GP models...")
    overpotential_model = CatalystGPModel('overpotential')
    overpotential_model.fit(X, y_overpotential)
    
    tafel_model = CatalystGPModel('tafel_slope')
    tafel_model.fit(X, y_tafel)
    
    # Setup optimization
    print("Setting up optimization...")
    optimizer = CatalystOptimizer(overpotential_model, tafel_model)
    
    feature_bounds, categorical_mappings = create_optimization_bounds()
    optimizer.setup_optimization(feature_bounds, categorical_mappings)
    
    # Run optimization
    print("Running NSGA-II optimization...")
    result = optimizer.optimize(n_generations=50)
    
    # Get results
    print("\nOptimization completed!")
    best_solution = optimizer.get_best_solution()
    print(f"Best solution:")
    print(f"  Dopant: {best_solution['dopant_material']}")
    print(f"  Concentration: {best_solution['doping_concentration']:.1f} wt.%")
    print(f"  Overpotential: {best_solution['overpotential']:.1f} mV")
    print(f"  Tafel Slope: {best_solution['tafel_slope']:.1f} mV/dec")
    
    # Plot results
    optimizer.plot_pareto_front()
    optimizer.plot_convergence()
    
    # Run sensitivity analysis
    print("\nRunning sensitivity analysis...")
    weight_combinations = [(0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]
    sensitivity_results = run_sensitivity_analysis(optimizer, weight_combinations)
    
    print("Sensitivity analysis results:")
    for weights, results in sensitivity_results.items():
        eta_w, ts_w = weights.split('_')
        print(f"  Weights η={eta_w}, TS={ts_w}: {results['best_dopant']} at {results['best_concentration']:.1f} wt.%")

