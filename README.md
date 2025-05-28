# learnify-ml

**learnify-ml** is a lightweight and modular Python package designed to **automate preprocessing pipelines** for machine learning datasets. With minimal configuration, it applies standard best practices to clean, transform, and prepare data for training.

---

## Features

### Data Preprocessing
-  Automatically detects and fills missing values
-  Identifies and handles outliers
-  Applies skewness correction to numeric features
-  Scales numerical data using standard scaling
-  Encodes categorical variables using label encoding
-  Removes low-variance features
-  Performs VIF analysis to reduce multicollinearity
-  Balances imbalanced datasets using SMOTE
-  Supports feature selection methods

### Model Training & Evaluation
-  Trains multiple machine learning models (e.g., Random Forest, XGBoost, Logistic Regression)
-  Automatically selects the best-performing model based on scoring metrics
-  Tracks performance with metrics like accuracy, F1-score
-  Saves trained model and evaluation report as artifacts

### Hyperparameter Optimization
-  Supports both **GridSearchCV** and **RandomizedSearchCV**
-  Configurable search space for each model
-  Automatically selects best hyperparameters and retrains final model
---

## Installation

```bash
pip install learnify-ml
```

If the package is not yet on PyPI, you can install locally:

```
git clone https://github.com/yourusername/learnify-ml.git
cd learnify-ml
pip install -e .
```

## Usage
🔹 1. Import the main pipeline class

```python
from learnify_ml import AutoMLPipeline
```

🔹 2. Run the pipeline

```python
trainer = AutoMLPipeline(data_path="data.csv", target_column="target")
trainer.run_pipeline()
```

## Example Functions

```python
df = trainer.remove_outliers(df)
df = trainer.handle_missing_values(df)
df = trainer.scale_numeric_features(df)
```

Each method is modular and can be used independently or inside run_pipeline().





