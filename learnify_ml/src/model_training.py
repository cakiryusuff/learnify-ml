import pandas as pd
from learnify_ml.src.model_training_engine import ModelTrainingEngine
from learnify_ml.utils.common_functions import load_data, save_data
from learnify_ml.src.custom_exception import CustomException
from learnify_ml.src.logger import get_logger
from learnify_ml.config.config_paths import *
from learnify_ml.config.model_config import default_models, default_params, default_search_methods, default_search_methods_params, default_metrics
from typing import Literal
import os
from tqdm import tqdm

from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, classification_report,
        mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
    )
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

from typing import Any

logger = get_logger(__name__)

class ModelTrainer:
    """
    A class to handle the training of machine learning models.
    
    This class includes methods for splitting data, training models, evaluating them, and saving the best model.
    
    Parameters
    ----------
    target_column (str) : The name of the target column in the data.
    data_path (str) : The path to the preprocessed data.
    model_save_path (str) : The path to save the trained models.
    model (sklearn model) : The default model to be trained if hyperparameter tuning is not applied.
    models_list (dict) : A dictionary of models to be used for Hyperparameter Tuning and trained (e.g., {"RandomForest": RandomForestClassifier(), "KNeighbors": KNeighborsClassifier()}).
    params_list (dict) : A dictionary of hyperparameters for each model {"RandomForest": {"n_estimators": [100, 200]}, "KNeighbors": {"n_neighbors": [3, 5]}}.
    apply_hyperparameter_tuning (bool) : Whether to apply hyperparameter tuning or not.
    use_case (str) : The type of machine learning task (e.g., "classification", "regression", "clustering").
    test_size (float) : The proportion of the data to use for testing.
    random_state (int) : The random state for reproducibility.
    hyperparameter_tuning_method (Literal["randomized", "grid"]) : The method to use for hyperparameter tuning.
    """
    def __init__(self,
                target_column: str,
                use_case: Literal["classification", "regression"] = "classification",  # e.g., classification, regression
                model = RandomForestClassifier(),
                models_list: dict = None,
                params_list: dict = None,
                apply_hyperparameter_tuning: bool = False,
                hyperparameter_tuning_method: Literal["randomized", "grid"] = "randomized",
                data_path: str = DATA_PREPROCESSING_OUTPUT,
                model_save_path: str = MODEL_SAVE_PATH,
                test_size: float = 0.2,
                random_state: int = 42,
                search_methods: dict[str, Any] = None,
                search_methods_params: dict[str, dict[str, dict[str, Any]]] = None
                ):
        
        self.target_column = target_column
        
        if self.target_column is None:
            raise ValueError("target_column must be specified")
        
        self.use_case = use_case
        self.model = model
        self.models = models_list or default_models
        self.params = params_list or default_params
        self.apply_hyperparameter_tuning = apply_hyperparameter_tuning
        self.hyperparameter_tuning_method = hyperparameter_tuning_method
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.test_size = test_size
        self.random_state = random_state
        self.is_multiclass = None
        self.search_methods = search_methods or default_search_methods
        self.search_methods_params = search_methods_params or default_search_methods_params
        self.default_metrics = default_metrics

        
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        
    def split_train_test(self, df: pd.DataFrame):
        """
        Split the DataFrame into training and testing sets.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data to be split.
        
        Returns:
        X_train, X_test, y_train, y_test: The training and testing sets.
        """
        try:
            logger.info("Splitting data into training and testing sets")
            X = df.drop(self.target_column, axis=1)
            y = df[self.target_column]
            
            if self.use_case == "classification":
                self.is_multiclass = "micro" if len(y.unique()) > 2 else "binary"
                
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error in splitting data: {e}")
            raise CustomException(e, "Error in splitting data")
        
    def train_models(self, X_train, y_train):
        try:
            logger.info("Model training process started")
            engine = ModelTrainingEngine(
                models=self.models,
                params=self.params,
                model=self.model,
                use_case=self.use_case,
                apply_hyperparameter_tuning=self.apply_hyperparameter_tuning,
                hyperparameter_tuning_method=self.hyperparameter_tuning_method,
                search_methods=self.search_methods,
                search_methods_params=self.search_methods_params,
                random_state=self.random_state
            )
            # elif self.use_case == "regression":
            #     engine = RegressionTrainEngine()
                

            return engine.train(X_train, y_train)

        except Exception as e:
            logger.error(f"Error in training models: {e}")
            raise CustomException(e, "Error in training models")

    def get_metrics(self, y_true, y_pred, average=None):
        """
        Calculate evaluation metrics based on the use case.

        Parameters:
        - y_true (array-like): True labels.
        - y_pred (array-like): Predicted labels.
        - average (str): Averaging method for classification metrics.

        Returns:
        - dict: Dictionary of relevant metrics.
        """
        try:
            if self.use_case == "classification":
                return {
                    "accuracy": accuracy_score(y_true, y_pred),
                    "f1_score": f1_score(y_true, y_pred, average=average),
                    "precision": precision_score(y_true, y_pred, average=average),
                    "recall": recall_score(y_true, y_pred, average=average),
                    "classification_report": classification_report(y_true, y_pred)
                }

            elif self.use_case == "regression":
                return {
                    "mean_squared_error": mean_squared_error(y_true, y_pred),
                    "root_mean_squared_error": root_mean_squared_error(y_true, y_pred),
                    "mean_absolute_error": mean_absolute_error(y_true, y_pred),
                    "r2_score": r2_score(y_true, y_pred)
                }

            else:
                raise ValueError(f"Unsupported use case: {self.use_case}")

        except Exception as e:
            logger.error(f"Error in calculating metrics: {e}")
            raise CustomException(e, "Error in calculating metrics")

        
    def evaluate_models(self, results, X_test, y_test):
        """
        Evaluate multiple models and save the evaluation results to CSV.

        Parameters:
        - results (dict): Dictionary with model names and their best estimators.
        - X_test (array-like): Test features.
        - y_test (array-like): Test target values.

        Returns:
        - dict: Dictionary containing metrics per model.
        """
        try:
            logger.info("Starting model evaluation...")

            evaluation_results = {}
            rows = []

            for name, result in results.items():
                best_model = result["best_model"]
                y_pred = best_model.predict(X_test)

                metrics = self.get_metrics(y_test, y_pred, average=self.is_multiclass)
                evaluation_results[name] = metrics

                if self.use_case == "classification":
                    row = {
                        "model": name,
                        "accuracy": metrics["accuracy"],
                        "f1_score": metrics["f1_score"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"]
                    }
                    logger.info(f"- Model {name}: Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")

                elif self.use_case == "regression":
                    row = {
                        "model": name,
                        "mean_squared_error": metrics["mean_squared_error"],
                        "root_mean_squared_error": metrics["root_mean_squared_error"],
                        "mean_absolute_error": metrics["mean_absolute_error"],
                        "r2_score": metrics["r2_score"]
                    }
                    logger.info(f"- Model {name}: RMSE: {metrics['root_mean_squared_error']:.4f}, MAE: {metrics['mean_absolute_error']:.4f}, R²: {metrics['r2_score']:.4f}")

                rows.append(row)

            data_csv = pd.DataFrame(rows)

            save_data(data_csv, self.model_save_path + "model_evaluation_results.csv")

            return evaluation_results

        except Exception as e:
            logger.error(f"Error in evaluating models: {e}")
            raise CustomException(e, "Error in evaluating models")

    
    def save_best_model(self, best_model, best_model_name):
        """
        Save the best model to the specified path.
        
        Parameters:
        best_model: The best trained model to be saved.
        """
        try:
            logger.info(f"Saving the best model to {self.model_save_path}")
            import joblib
            joblib.dump(best_model, self.model_save_path + f'best_model_{best_model_name}.pkl'.lower().strip())
            logger.info("- Best model saved successfully")
        except Exception as e:
            logger.error(f"Error saving the best model: {e}")
            raise CustomException(e, "Error saving the best model")
    
    def run_training(self):
        try:
            logger.info("------------------------------------------------------")            
            df = load_data(self.data_path)
            
            X_train, X_test, y_train, y_test = self.split_train_test(df)
            
            results = self.train_models(X_train, y_train)
            
            evaluation_results = self.evaluate_models(results, X_test, y_test)
            
            best_model_name = max(evaluation_results, key=lambda x: evaluation_results[x][self.default_metrics[self.use_case]])
                
            best_model = results[best_model_name]["best_model"]
            self.save_best_model(best_model, best_model_name)
            
            logger.info("Model training process completed successfully")
            
        except Exception as e:
            logger.error(f"Error in model training process: {e}")
            raise CustomException(e, "Error in model training process")
        
if __name__ == "__main__":
    trainer = ModelTrainer(target_column="Survived")
    trainer.run_training()