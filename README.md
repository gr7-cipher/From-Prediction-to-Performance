# Machine Learning-Guided Discovery of Cu-Doped Co₃O₄ Electrocatalysts

This repository contains the complete code implementation for the research paper "From Prediction to Performance: Machine Learning-Guided Discovery and Validation of Cu-Doped Co₃O₄ for Enhanced Oxygen Evolution Reaction".

## Overview

This work presents a novel AI-driven framework that leverages machine learning and multi-objective optimization to design high-performance, metal-doped Co₃O₄ catalysts for the oxygen evolution reaction (OER). The framework combines Gaussian Process (GP) regression models with NSGA-II genetic algorithm optimization.

## Repository Structure

```
ml_catalyst_discovery/
├── src/
│   ├── data_preprocessing.py          # Data loading and preprocessing
│   ├── gaussian_process_models.py     # GP model implementation
│   ├── nsga_ii_optimization.py        # Multi-objective optimization
│   ├── generate_figures.py            # Figure generation utilities
│   ├── main_analysis.py               # Complete analysis workflow
│   ├── run_quick_analysis.py          # Quick analysis for key results
│   └── create_readme.py               # This file
├── data/                              # Data files (to be added)
├── results/                           # Generated results and figures
├── notebooks/                         # Jupyter notebooks (optional)
└── docs/                             # Documentation
```

## Key Features

### 1. Gaussian Process Models
- Composite kernel implementation (Matern + White Noise)
- Hyperparameter optimization using L-BFGS-B algorithm
- Cross-validation and uncertainty quantification
- Feature importance analysis using permutation importance

### 2. Multi-Objective Optimization
- NSGA-II algorithm implementation
- Simultaneous optimization of overpotential and Tafel slope
- Pareto front generation and analysis
- Sensitivity analysis for different weight combinations

### 3. Data Analysis
- Comprehensive data preprocessing pipeline
- Statistical significance testing for correlations
- Learning curve analysis for model validation
- Kernel comparison (Matern vs RBF)

## Installation

### Requirements
- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- SciPy
- pymoo (for NSGA-II optimization)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd ml_catalyst_discovery

# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn scipy pymoo

# Run the analysis
cd src
python run_quick_analysis.py
```

## Usage

### Quick Analysis
To generate the key figures and tables required for the manuscript:

```bash
cd src
python run_quick_analysis.py
```

This will generate:
- Supplementary Figure S1: Correlation matrix with p-values
- Supplementary Figure S5: Learning curves (R² vs training set size)
- Supplementary Table S2: Cross-validation results
- Feature importance analysis
- Kernel comparison results
- Complete dataset (Supplementary Data 1)

### Full Analysis
For the complete analysis including optimization:

```bash
cd src
python main_analysis.py
```

### Individual Components

#### Data Preprocessing
```python
from data_preprocessing import CatalystDataProcessor

processor = CatalystDataProcessor()
df = processor.load_literature_data()
X, y, report = processor.preprocess_data(df)
```

#### GP Model Training
```python
from gaussian_process_models import CatalystGPModel

model = CatalystGPModel('overpotential')
model.fit(X, y)
cv_results = model.cross_validate(X, y)
```

#### Multi-Objective Optimization
```python
from nsga_ii_optimization import CatalystOptimizer

optimizer = CatalystOptimizer(overpotential_model, tafel_model)
optimizer.setup_optimization(feature_bounds, categorical_mappings)
result = optimizer.optimize(n_generations=50)
```

## Key Results

### Model Performance (Cross-Validation)
- **Overpotential Model**: R² = 0.92 ± 0.04, RMSE = 0.08 ± 0.02 V
- **Tafel Slope Model**: R² = 0.88 ± 0.06, RMSE = 12.1 ± 2.5 mV/dec

### Optimization Results
- **Best Catalyst**: Cu-doped Co₃O₄ at 44 wt.%
- **Predicted Performance**: η = 236 ± 15 mV, TS = 81 ± 5 mV/dec
- **Experimental Validation**: η = 240 mV, TS = 83.2 mV/dec

### Kernel Comparison
- **Matern Kernel**: Superior log-marginal likelihood and lower prediction errors
- **Statistical Validation**: All reported features significant (p < 0.05)

##  Files

### Figures
- `supplementary_figure_s1_correlation_pvalues.png`: Correlation matrix with statistical significance
- `supplementary_figure_s5_learning_curve_overpotential.png`: Learning curve for overpotential model
- `supplementary_figure_s5_learning_curve_tafel.png`: Learning curve for Tafel slope model
- `feature_importance_overpotential.png`: Feature importance for overpotential
- `feature_importance_tafel_slope.png`: Feature importance for Tafel slope
- `kernel_comparison_overpotential.png`: Kernel comparison for overpotential
- `kernel_comparison_tafel_slope.png`: Kernel comparison for Tafel slope

### Tables
- `supplementary_table_s2_cross_validation.csv`: Cross-validation metrics
- `kernel_comparison_results.csv`: Detailed kernel comparison
- `feature_importance_results.csv`: Feature importance with statistical significance

### Data
- `complete_dataset_supplementary_data_1.csv`: Complete experimental dataset
- `processed_features.csv`: Preprocessed feature matrix
- `processed_targets.csv`: Preprocessed target variables

## Methodology

### 1. Data Preprocessing
- Literature data compilation (95 samples)
- Missing value handling (Tafel slope removal)
- One-hot encoding for categorical variables
- Log-transformation for overpotential values

### 2. Gaussian Process Modeling
- Composite kernel: k(x,x') = k_C(x,x') * k_M(x,x') + k_WN(x,x')
- Hyperparameter optimization via log-marginal likelihood maximization
- 10-fold cross-validation for model validation

### 3. Feature Importance
- Permutation importance with 50 repetitions
- Statistical significance testing (one-sample t-test)
- Reproducibility ensured through random seed control

### 4. Multi-Objective Optimization
- NSGA-II algorithm (50 generations, population size 50)
- Composite objective: 0.7 * η_norm + 0.3 * TS_norm
- Pareto front analysis and convergence monitoring

## Citation

If you use this code in your research, please cite:

```bibtex
@article{catalyst_discovery_2025,
  title={From Prediction to Performance: Machine Learning-Guided Discovery and Validation of Cu-Doped Co₃O₄ for Enhanced Oxygen Evolution Reaction},
  author={Al-Bataineh, Qais M. and Ahmad, Ahmad A. and Bani-Salameh, Areen A. and others},
  journal={International Journal of Hydrogen Energy},
  year={2025}
}
```

## License

This project is licensed under the Apache 2.0 License License - see the LICENSE file for details.

## Contact

For questions or collaborations, please contact:
- Qais M. Al-Bataineh: qais.albataineh@tu-dortmund.de
- Ahmad A. Ahmad: sema@just.edu.jo

## Acknowledgments

This work was supported by [funding agencies and institutions]. We thank the reviewers for their valuable feedback that improved the quality of this research.
