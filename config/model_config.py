from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform

models = {
    "RandomForest": RandomForestClassifier(),
    "KNeighbors": KNeighborsClassifier(),
    "SVC": SVC(),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "LightGBM": LGBMClassifier(verbosity=-1)
}

params = {
    "RandomForest": {
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
    }
}
