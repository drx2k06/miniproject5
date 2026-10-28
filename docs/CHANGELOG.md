# Changelog

All notable changes to the **House Price Prediction System** (`mini-project5`) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2026-10-28 (Modernized Engineering & Quality Release)

### Added
- **Core Architecture (`src/`)**:
  - `src/config.py`: Centralized configuration, immutable paths, schema definitions, and validation limits.
  - `src/data_loader.py`: Safe dataset loading, schema verification, missing data checks, and stratified train/test split.
  - `src/preprocessing.py`: Scikit-learn Pipeline with `ColumnTransformer` (StandardScaler for numerics, OneHotEncoder for categoricals).
  - `src/models.py`: Multi-model training and evaluation harness comparing Linear Regression, Ridge, Lasso, Random Forest, and Gradient Boosting with leak-free CV selection.
  - `src/evaluation.py`: Regression evaluation metrics suite ($\text{MAE}$, $\text{RMSE}$, $\text{MAPE}$, $R^2$, $\text{MedAE}$), residual analysis, and Prediction Interval Coverage Probability (PICP).
  - `src/predictor.py`: Enterprise-grade `HousePricePredictor` with domain sanity checks, floor clamping (preventing negative valuations), 95% prediction intervals, and legacy 4-feature backwards compatibility.
  - `src/api.py`: High-performance FastAPI REST API microservice with Pydantic validation schemas.
- **Dataset (`data/`)**:
  - `data/house_prices.csv`: 600-sample realistic residential housing dataset across Indian property tiers with physically coherent relationships between area, bedrooms, bathrooms, age, location, and price (explicitly disclosed as parametric synthetic benchmark).
- **CLI & Automation (`scripts/`)**:
  - `scripts/train.py`: CLI training script comparing all models, selecting the champion, and serializing model artifacts with SHA-256 metadata.
  - `scripts/evaluate.py`: Standalone evaluation script outputting performance tables and residual diagnostics.
- **Automated Test Suite (`tests/`)**:
  - `tests/test_api.py`: 9 unit tests for FastAPI REST endpoints (health, predict, batch, domain and range validation).
  - `tests/test_data_loader.py`: 6 unit tests for data loading and validation.
  - `tests/test_preprocessing.py`: 3 unit tests for pipeline transformations.
  - `tests/test_models.py`: 3 unit tests for model training, serialization, and metadata hashing.
  - `tests/test_predictor.py`: 7 unit tests for inference, boundary handling, negative price clamping, and feature importances.
  - `tests/test_app_integration.py`: 2 integration tests for Streamlit UI helpers and prediction pipelines.
- **Documentation (`docs/`)**:
  - `docs/REPOSITORY_AUDIT.md`: Forensic audit of the baseline codebase.
  - `docs/ARCHITECTURE.md`: Technical architectural specification with Mermaid diagrams.
  - `docs/IMPROVEMENT_ROADMAP.md`: Structured roadmap of milestones and deliverables.
  - `docs/EVALUATION_REPORT.md`: Rigorous comparative evaluation of baseline vs. final models.
  - `docs/FINAL_REPORT.md`: Comprehensive engineering handoff document.
- **Project Hygiene**:
  - `.gitignore`: Standardized, comprehensive Git ignore rules replacing the misnamed file.
  - `LICENSE`: MIT License preserving original authorship attribution.
  - `pyproject.toml`: Modern packaging and test configuration.
  - `requirements-dev.txt`: Development and testing dependencies.
  - `.env.example`: Safe environment configuration template.

### Changed
- **`app.py`**:
  - Upgraded from a 60-line basic script to an interactive multi-tab Streamlit dashboard:
    - Tab 1: Single Property Valuation with instant metrics, confidence intervals, and price per sqft.
    - Tab 2: Model Performance & Explainability displaying comparative benchmarks and feature importance.
    - Tab 3: Batch CSV Valuation with download support.
    - Tab 4: Dataset Explorer with dynamic charts and correlations.
  - Added defensive error handling for missing models with clear instructions to train.
- **`train_model.py`**:
  - Refactored to act as a backwards-compatible wrapper calling the modernized modular training pipeline.
- **`README.md`**:
  - Completely rewritten with verified installation instructions, Windows PowerShell examples, architecture guide, and evaluation results.
- **`requirements.txt`**:
  - Added clean dependency specifications.

### Fixed
- **Inverted Regression Coefficients**: Resolved multicollinearity and small-sample instability by training on realistic data with regularized/ensemble models.
- **Negative Price Prediction Anomaly**: Enforced physical floor clamping and log-scale bounded estimation so houses cannot be evaluated with negative prices.
- **Statistically Degenerate Evaluation**: Replaced 2-sample test evaluation with 5-fold cross-validation and independent hold-out test set ($N=100$).
- **Misnamed Gitignore**: Renamed `House-Price-Prediction.gitignore` to `.gitignore`.

---

## [1.0.0] - 2026-10-26 (Original Baseline Repository)

### Initial State
- Initial commit by kirankumarnscr7-cmyk.
- 10-row synthetic dataset in `train_model.py`.
- LinearRegression model saved to `house_price_model.pkl`.
- Basic Streamlit app in `app.py`.
- Misnamed `House-Price-Prediction.gitignore`.
- No automated tests or modular package structure.
