import pandas as pd
from utils.common_functions import load_data, save_data
from src.custom_exception import CustomException
from src.logger import get_logger
from config.config_paths import *
from config.model_config import models, params
import os

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

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
    """
    def __init__(self, target_column: str = ...,
                 data_path: str = DATA_PREPROCESSING_OUTPUT,
                 model_save_path: str = MODEL_SAVE_PATH,
                 model = RandomForestClassifier(),
                 models_list: dict = models,
                 params_list: dict = params,
                 apply_hyperparameter_tuning: bool = False,
                 use_case: str = "classification", # TODO: Regression and clustering will be added
                 test_size: float = 0.2,
                 random_state: int = 42):
        
        self.target_column = target_column
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.model = model
        self.models = models_list
        self.params = params_list
        self.apply_hyperparameter_tuning = apply_hyperparameter_tuning
        self.use_case = use_case
        self.test_size = test_size
        self.random_state = random_state
        
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        
    def split_train_test(self, df: pd.DataFrame):
        # TODO: Add more robust splitting logic if needed (e.g., stratified split)
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
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error in splitting data: {e}")
            raise CustomException(e, "Error in splitting data")
        
    def train_models(self, X_train, y_train) -> dict:
        # TODO: Add more models and hyperparameters as needed
        try:
            if self.apply_hyperparameter_tuning:
                logger.info("Starting model training")
                results = {}
                for name, model in models.items():
                    logger.info(f"Training model: {name}")
                    search = RandomizedSearchCV(model, params[name], n_iter=10, cv=3, scoring='accuracy', random_state=self.random_state)
                    search.fit(X_train, y_train)
                    results[name] = {
                        "best_model": search.best_estimator_,
                        "best_score": search.best_score_,
                        "best_params": search.best_params_
                    }
                    logger.info(f"Model {name} trained with best score: {search.best_score_}")
                return results
            else:
                logger.info(f"Starting Single Model Training {str(self.model)}")
                results = {"single_model": {"best_model": self.model, "best_score": None, "best_params": None}}
                self.model.fit(X_train, y_train)
                return results
                
        except Exception as e:
            logger.error(f"Error in training models: {e}")
            raise CustomException(e, "Error in training models")
        
    def evaluate_models(self, results, X_test, y_test):
        try:
            logger.info("Starting model evaluation")
            evaluation_results = {}
            for name, result in results.items():
                best_model = result["best_model"]
                y_pred = best_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                report = classification_report(y_test, y_pred)
                
                evaluation_results[name] = {
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall,
                    "classification_report": report
                }
                
                logger.info(f"- Model {name}: Accuracy: {accuracy}, F1 Score: {f1}, Precision: {precision}, Recall: {recall}")
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
            logger.info("Starting model training process")
            
            df = load_data(self.data_path)
            X_train, X_test, y_train, y_test = self.split_train_test(df)
            results = self.train_models(X_train, y_train)
            evaluation_results = self.evaluate_models(results, X_test, y_test)
            
            best_model_name = max(evaluation_results, key=lambda x: evaluation_results[x]['accuracy'])
            best_model = results[best_model_name]["best_model"]
            self.save_best_model(best_model, best_model_name)
            
            # logger.info(f"Best model: {best_model_name} with accuracy: {evaluation_results[best_model_name]['accuracy']}")
            logger.info("Model training process completed successfully")
            
        except Exception as e:
            logger.error(f"Error in model training process: {e}")
            raise CustomException(e, "Error in model training process")
        
if __name__ == "__main__":
    trainer = ModelTrainer(target_column="Survived")
    trainer.run_training()