from learnify_ml.config.config_paths import *
from learnify_ml.config.model_config import models, params
from learnify_ml.src.data_preprocessing import DataPreprocessor
from learnify_ml.src.model_training import ModelTrainer
from typing import Literal
from sklearn.ensemble import RandomForestClassifier

class AutoMLPipeline:
    """Configuration class for the AutoML pipeline.
    
    This class holds the parameters for the AutoML pipeline including data preprocessing, model configurations, and hyperparameters.
    
    Parameters
    ----------
    
    target_column (str) : The name of the target column in the dataset.
    data_path (str) : The path to the input dataset.
    data_output_path (str) : The path where the preprocessed dataset will be saved.
    impute_strategy (str) : The strategy to use for imputation (e.g., 'mean', 'median', 'most_frequent').
    impute_strategy_remove (str) : The strategy to use for imputation (e.g., 'column', 'row', None).
    apply_smote (bool) : Whether to apply SMOTE (Synthetic Minority Over-sampling Technique) or not.
    apply_scale (bool) : Whether to apply scaling or not.
    apply_outlier (bool) : Whether to apply outlier detection or not.
    apply_vif (bool) : Whether to apply VIF (Variance Inflation Factor) or not.
    apply_skewness (bool) : Whether to apply skewness detection or not.
    encode_target_column (bool): Whether to encode the target column or not.
    model_save_path (str) : The path to save the trained models.
    model (sklearn model) : The default model to be trained if hyperparameter tuning is not applied.
    models_list (dict) : A dictionary of models to be used for Hyperparameter Tuning and trained (e.g., {"RandomForest": RandomForestClassifier(), "KNeighbors": KNeighborsClassifier()}).
    params_list (dict) : A dictionary of hyperparameters for each model {"RandomForest": {"n_estimators": [100, 200]}, "KNeighbors": {"n_neighbors": [3, 5]}}.
    apply_hyperparameter_tuning (bool) : Whether to apply hyperparameter tuning or not.
    hyperparameter_tuning_method (str) : The method to use for hyperparameter tuning (e.g., "randomized", "grid").
    use_case (str) : The type of machine learning task (e.g., "classification", "regression", "clustering").
    test_size (float) : The proportion of the data to use for testing.
    random_state (int) : The random state for reproducibility.
    """
    def __init__(self,
                target_column: str = ...,
                use_case: str = "classification",
                test_size: float = 0.2,
                random_state: int = 42,

                data_path: str | None = DATA_PREPROCESSING_INPUT,
                data_output_path: str | None = DATA_PREPROCESSING_OUTPUT,
                model_save_path: str = MODEL_SAVE_PATH,
                
                impute_strategy: Literal["mean", "median", "most_frequent"] = "mean",
                impute_strategy_remove: Literal["column", "row", None] = None,
                apply_outlier: bool = False,
                apply_vif: bool = False,
                apply_skewness: bool = False,
                apply_scale: bool = True,
                apply_smote: bool = True,
                encode_target_column: bool = False,
                
                model = RandomForestClassifier(),
                models_list: dict = models,
                params_list: dict = params,
                apply_hyperparameter_tuning: bool = False,
                hyperparameter_tuning_method: Literal["randomized", "grid"] = "randomized"
                ):
        
        self.target_column = target_column
        self.data_path = data_path
        self.data_output_path = data_output_path
        self.impute_strategy = impute_strategy
        self.impute_strategy_remove = impute_strategy_remove
        self.apply_smote = apply_smote
        self.apply_scale = apply_scale
        self.apply_outlier = apply_outlier
        self.apply_vif = apply_vif
        self.apply_skewness = apply_skewness
        self.apply_hyperparameter_tuning = apply_hyperparameter_tuning
        self.use_case = use_case
        self.test_size = test_size
        self.random_state = random_state
        self.model_save_path = model_save_path
        self.model = model
        self.models_list = models_list
        self.params_list = params_list
        self.encode_target_column = encode_target_column
        self.hyperparameter_tuning_method = hyperparameter_tuning_method
        
    def run_pipeline(self):
        """
        Runs the entire AutoML pipeline including data preprocessing and model training.
        """
        preprocessor = DataPreprocessor(self.target_column,
                                        self.data_path,
                                        self.data_output_path,
                                        self.impute_strategy,
                                        self.impute_strategy_remove,
                                        self.apply_smote,
                                        self.apply_scale,
                                        self.apply_outlier,
                                        self.apply_vif,
                                        self.apply_skewness,
                                        self.encode_target_column
                                        )
        preprocessor.run_preprocessing()

        trainer = ModelTrainer(self.target_column,
                               self.data_output_path,
                               self.model_save_path,
                               self.model,
                               self.models_list,
                               self.params_list,
                               self.apply_hyperparameter_tuning,
                               self.use_case,
                               self.test_size,
                               self.random_state,
                               self.hyperparameter_tuning_method
                               )
        trainer.run_training()