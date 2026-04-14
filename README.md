# House Price Prediction with Machine Learning Ensemble

A supervised machine learning project that predicts residential house sale prices using an ensemble of regression models, with GPU acceleration support.

## Project Overview

| Item | Detail |
|------|--------|
| **Dataset** | Kaggle Ames Housing Dataset |
| **Training Samples** | 1,460 |
| **Features** | 81 raw features (numerical + categorical) |
| **Target** | `SalePrice` (skewed, log-transformed for training) |
| **Environment** | Python 3, Jupyter Notebook |
| **GPU** | NVIDIA RTX 4080 (CUDA 12.x) |

## Models & Methods

### Data Preprocessing
- **Numerical**: SimpleImputer (mean fill) + StandardScaler
- **Categorical**: SimpleImputer (constant fill) + OneHotEncoder
- **Pipeline**: sklearn `Pipeline` + `ColumnTransformer`

### Feature Engineering
- `PropertyAge` = YrSold − YearBuilt
- `TotalSF` = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
- `TotalBath` = FullBath + 0.5 × HalfBath + BsmtFullBath + 0.5 × BsmtHalfBath
- `HasRemodeled`, `Has2ndFloor`, `HasGarage` (binary flags)
- **PCA**: Retains 95% of variance to reduce dimensionality

### Models (with GridSearchCV, 3-fold CV)

| Model | Device | RMSE (test) |
|-------|--------|:-----------:|
| Linear Regression | CPU | 0.142 |
| Random Forest | CPU | 0.153 |
| **XGBoost** | **GPU (CUDA)** | **0.138** |
| MLP (Neural Network) | CPU | 0.135 |
| Average Ensemble (RF + XGB + MLP) | — | — |
| **Stacking Ensemble** | — | **0.134** |

### Stacking Ensemble
- Base learners: LinearRegression, RandomForest, XGBoost, MLP
- Meta-learners tested: MLP, LinearRegression, XGBoost
- Best meta-learner: XGBoost

## Project Structure

```
ML-House_Price_Predict/
├── house-price-predict.ipynb    # Main notebook (all code + comments)
├── house-price-predict_picture.ipynb  # Notebook with pre-run outputs
├── kaggle/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
└── submission*.csv              # Model prediction outputs
```

## Key Findings

- **GrLivArea** (above-grade living area) has the strongest positive correlation with sale price (r = 0.71)
- **PropertyAge** has a moderate negative correlation with sale price (r = −0.52)
- XGBoost consistently outperforms Random Forest and Linear Regression across all feature sets
- PCA reduces dimensionality significantly while maintaining model performance
- Stacking ensemble provides marginal improvement over the best individual model (XGBoost)
- GPU acceleration reduces XGBoost training time substantially compared to CPU-only training

## Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn xgboost scipy plotly
```

### Run

Open `house-price-predict.ipynb` in Jupyter Notebook and run all cells sequentially.

To enable GPU acceleration for XGBoost, ensure CUDA is available:

```python
import torch
CUDA_AVAILABLE = torch.cuda.is_available()
# XGBoost will automatically use 'cuda' device if available
```

## Results

| Metric | Value |
|--------|-------|
| Best Individual Model | XGBoost (RMSE: 0.138) |
| Best Ensemble | Stacking (RMSE: 0.134) |
| Submission File | `submission.csv` |

## Author

Dylan Zhou
