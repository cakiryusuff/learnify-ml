from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

default_search_methods_params = {
    "randomized": {
        "n_iter": 10,
        "cv": 5,
        "scoring": "accuracy",
        "random_state": 42
    },
    "grid": {
        "cv": 5,
        "scoring": "accuracy"
    }
}

default_search_methods = {
    "randomized": RandomizedSearchCV,
    "grid": GridSearchCV
}


default_models = {
    "RandomForest": RandomForestClassifier(),
    "KNeighbors": KNeighborsClassifier(),
    "SVC": SVC(),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "LightGBM": LGBMClassifier(verbosity=-1)
}

default_params = {
    "grid": {
        "RandomForest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15, 20],
            "min_samples_split": [2, 5, 10]
        },
        "KNeighbors": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"]
        },
        "SVC": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        },
        "XGBoost": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2]
        },
        "LightGBM": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2]
        }
    },
    "randomized": {
        "RandomForest":{
        "n_estimators": randint(50, 200),
        "max_depth": randint(5, 20),
        "min_samples_split": randint(2, 10)
    },
    "KNeighbors": {
        "n_neighbors": randint(3, 15),
        "weights": ["uniform", "distance"]
    },
    "SVC": {
        "C": uniform(0.1, 10),
        "kernel": ["linear", "rbf"]
    },
    "XGBoost": {
        "n_estimators": randint(50, 200),
        "max_depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.3)
    },
    "LightGBM": {
        "n_estimators": randint(50, 200),
        "max_depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.3)
    }}
}
