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

## Pipeline Usage
🔹 1. Import the main pipeline class

Main Pipeline Class contains DataPreprocessor and ModelTrainer

```python
from learnify_ml import AutoMLPipeline
```

🔹 2. Run the pipeline

```python
trainer = AutoMLPipeline(target_column="target_column",
                      use_case="regression",
                      apply_hyperparameter_tuning=True,
                      hyperparameter_tuning_method="randomized",
                      apply_tf_idf=False,
                      apply_scale=True,
                      apply_feature_selection=True,
                      apply_outlier=True,
                      apply_vif=True,
                      apply_skewness=True,
                      apply_smote=False,
                      test_size=0.2,
                      impute_strategy="mean",
                      ).run_pipeline()

best_model, evaluation_results, df = trainer.run_pipeline()
```
## Visualizer
```python
from learnify_ml import DataVisualizer

data_visualizer = DataVisualizer(
    data_path='learnify_ml/artifacts/dataset/possum.csv'
)

data_visualizer.visualize_feature_distribution()

data_visualizer.visualize_correlation_matrix()

data_visualizer.visualize_pairwise_relationships_numerical('totlngth')
```


## DataPreprocessor

Each method is modular and can be used independently.

```python
from learnify_ml import DataPreprocessor
```

```python
data_preprocessor = DataPreprocessor(
    data_path='learnify_ml/artifacts/dataset/possum.csv',
    use_case="regression",
    target_column="totlngth",
    apply_outlier = False,
    apply_vif= False,
    apply_skewness = False,
    apply_tf_idf = False,
    encode_target_column = False,
    apply_scale = True,
    apply_smote = True,
    apply_feature_selection = True,
)

df = data_preprocessor.run_preprocessing()
```

or

```python
data_preprocessor = DataPreprocessor()

df = data_preprocessor.label_encode(
    df=df,
    categorical_columns=categorical_columns,
    target_column=target_column,
    encode_target_column=encode_target_column
)

df = data_preprocessor.variance_inflation(df=df)

df = data_preprocessor.split_object_columns(
    df=df,
    max_unique_thresh=30,
    min_avg_len_thresh=20
)
```

## ModelTrainer

You can pass your own models or use default RandomForest.

```python
from learnify_ml import ModelTrainer

model_trainer = ModelTrainer(
    data_path='learnify_ml/artifacts/dataset/possum.csv',
    use_case="regression",
    target_column="totlngth",
    apply_hyperparameter_tuning=True,
    hyperparameter_tuning_method="randomized"
)

best_model, evaluation_results = model_trainer.run_training()
```
ModelTrainer returns 2 value; best_model, evaluation_results.
