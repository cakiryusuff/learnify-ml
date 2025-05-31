from learnify_ml.src.custom_exception import CustomException
from learnify_ml.src.logger import get_logger
from learnify_ml.config.model_config import default_metrics
from sklearn.model_selection import KFold, cross_val_score
from tqdm import tqdm
from typing import Any

logger = get_logger(__name__)

class ModelTrainingEngine:
    def __init__(self, 
                 models,
                 params,
                 model,
                 use_case: str = "classification",
                 hyperparameter_tuning_method: str = "randomized",
                 apply_hyperparameter_tuning: bool = True,
                 search_methods: dict[str, Any] = None,
                 search_methods_params: dict[str, dict[str, dict[str, Any]]] = None,
                 random_state: int = 42):
        
        self.models = models
        self.params = params
        self.model = model
        self.use_case = use_case
        self.hyperparameter_tuning_method = hyperparameter_tuning_method
        self.apply_hyperparameter_tuning = apply_hyperparameter_tuning
        self.search_methods = search_methods
        self.search_methods_params = search_methods_params
        self.random_state = random_state
    
    def train(self, X_train, y_train) -> dict:
        try:
            results = {}

            if self.apply_hyperparameter_tuning:
                logger.info(f"Starting model training with {self.hyperparameter_tuning_method} hyperparameter tuning")
                for name, model in tqdm(self.models[self.use_case].items(), desc="Training Models with Hyperparameter Tuning"):
                    logger.info(f"- Training model: {name}")
                    search_class = self.search_methods[self.hyperparameter_tuning_method]
                    search_class_params = self.search_methods_params[self.use_case][self.hyperparameter_tuning_method]
                    search = search_class(model,
                                          param_distributions=self.params[self.use_case][self.hyperparameter_tuning_method][name],
                                          **search_class_params,
                                        #   n_jobs=-1
                                          )
                    search.fit(X_train, y_train)

                    results[name] = {
                        "best_model": search.best_estimator_,
                        "best_score": search.best_score_,
                        "best_params": search.best_params_
                    }
                    logger.info(f"- - Model {name} trained with best score: {search.best_score_}")

            elif self.apply_hyperparameter_tuning is False and self.model is not None:
                logger.info(f"Starting single model training with K-Fold Cross Validation: {str(self.model)}")
                kfold = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
                scores = cross_val_score(self.model, X_train, y_train, cv=kfold, scoring=default_metrics[self.use_case], n_jobs=-1)
                logger.info(f"- Cross-Validation Accuracy Scores: {scores}")
                logger.info(f"- Mean Accuracy: {scores.mean():.4f}")

                if hasattr(self.model, 'n_jobs'):
                    self.model.set_params(n_jobs=-1)

                with tqdm(total=1, desc="Training Single Model") as pbar:
                    self.model.fit(X_train, y_train)
                    pbar.update(1)
                
                results["single_model"] = {
                    "best_model": self.model,
                    "best_score": scores.mean(),
                    "best_params": None
                }
            else:
                logger.info("No hyperparameter tuning or model specified, skipping training.")
                raise CustomException("No hyperparameter tuning or model specified", "ModelTrainingEngine")

            return results

        except Exception as e:
            logger.error(f"Error in training models: {e}")
            raise CustomException(e, "Error in training models")
        