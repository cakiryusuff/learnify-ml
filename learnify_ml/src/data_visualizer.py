from learnify_ml.src.custom_exception import CustomException
from learnify_ml.src.logger import get_logger
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from learnify_ml.utils import load_data
import math

logger = get_logger(__name__)

class DataVisualizer:
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def visualize_missing_values(self):
        try:
            df = load_data(self.data_path)
            logger.info(f"Data loaded from {self.data_path}")
            missing_values = df.isnull().sum()
            plt.figure(figsize=(10, 6))
            sns.barplot(x=missing_values.index, y=missing_values.values, palette='viridis')
            plt.title('Missing Values Heatmap')
            plt.show()

        except Exception as e:
            raise CustomException(f"Error in visualizing missing values: {str(e)}")
        
    def visualize_feature_distribution(self):
        try:
            df = load_data(self.data_path)
            logger.info(f"Data loaded from {self.data_path}")

            df.hist(figsize=(12, 10), bins=30, edgecolor='black')
            plt.suptitle('Feature Distributions')
            plt.show()

        except Exception as e:
            raise CustomException(f"Error in visualizing feature distribution: {str(e)}")
    
    def visualize_target_distribution(self, target_column: str):
        try:
            df = load_data(self.data_path)
            logger.info(f"Data loaded from {self.data_path}")

            if target_column not in df.columns:
                raise CustomException(f"Target column '{target_column}' not found in the dataset.")

            plt.figure(figsize=(10, 6))
            sns.countplot(x=target_column, data=df)
            plt.title(f'Distribution of Target Variable: {target_column}')
            plt.show()

        except Exception as e:
            raise CustomException(f"Error in visualizing target distribution: {str(e)}")
    
    def visualize_feature_importance(self, model, feature_names):
        try:
            if not hasattr(model, 'feature_importances_'):
                raise CustomException("The provided model does not have feature importances.")

            importances = model.feature_importances_

            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
            plt.title('Feature Importance')
            plt.show()

        except Exception as e:
            raise CustomException(f"Error in visualizing feature importance: {str(e)}")
        
    def visualize_correlation_matrix(self):
        try:
            df = load_data(self.data_path)
            logger.info(f"Data loaded from {self.data_path}")

            plt.figure(figsize=(12, 8))
            sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
            plt.title('Correlation Matrix')
            plt.show()

        except Exception as e:
            raise CustomException(f"Error in visualizing correlation matrix: {str(e)}")
        
    def visualize_outliers(self, column: str):
        try:
            df = load_data(self.data_path)
            logger.info(f"Data loaded from {self.data_path}")

            if column not in df.columns:
                raise CustomException(f"Column '{column}' not found in the dataset.")

            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[column])
            plt.title(f'Boxplot of {column}')
            plt.show()

        except Exception as e:
            raise CustomException(f"Error in visualizing outliers: {str(e)}")

    def visualize_univariate_numerical(self, plots_per_page=4):
        df = pd.read_csv(self.data_path)

        num_features = df.select_dtypes(include=[np.number]).columns.tolist()

        if not num_features:
            raise Exception("No numerical features found in the dataset.")

        total = len(num_features)
        total_pages = math.ceil(total / plots_per_page)

        for page in range(total_pages):
            start = page * plots_per_page
            end = start + plots_per_page
            current_columns = num_features[start:end]

            nrows = len(current_columns)
            fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(10, nrows * 3))

            if nrows == 1:
                axes = np.expand_dims(axes, axis=0)

            for i, column in enumerate(current_columns):
                sns.histplot(data=df, x=column, ax=axes[i][0], kde=True)
                axes[i][0].set_title(f"Histogram for {column}")

                sns.boxplot(data=df, x=column, ax=axes[i][1])
                axes[i][1].set_title(f"Boxplot for {column}")

            plt.suptitle(f"Univariate Analysis - Page {page+1} / {total_pages}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

    def visualize_univariate_categorical(self, plots_per_page=4):
        df = pd.read_csv(self.data_path)

        cat_features = df.select_dtypes(include=['category']).columns.tolist()

        if not cat_features:
            raise Exception("No categorical features found in the dataset.")

        total = len(cat_features)
        total_pages = math.ceil(total / plots_per_page)

        for page in range(total_pages):
            start = page * plots_per_page
            end = start + plots_per_page
            current_columns = cat_features[start:end]

            nrows = len(current_columns)
            fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(10, nrows * 3))

            if nrows == 1:
                axes = np.expand_dims(axes, axis=0)

            for i, column in enumerate(current_columns):
                sns.countplot(data=df, x=column, ax=axes[i])
                axes[i].set_title(f"Count Plot for {column}")

            plt.suptitle(f"Univariate Analysis - Page {page+1} / {total_pages}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
    
    def visualize_pairwise_relationships_numerical(self, target_column, plots_per_page=4):
        df = pd.read_csv(self.data_path)

        num_features = df.select_dtypes(include=[np.number]).columns.tolist()

        if target_column not in df.columns:
            raise Exception(f"Target column '{target_column}' not found in the dataset.")

        if target_column in num_features:
            num_features.remove(target_column)

        if len(num_features) < 1:
            raise Exception("Not enough numerical features to visualize pairwise relationships.")

        total = len(num_features)
        total_pages = math.ceil(total / plots_per_page)

        for page in range(total_pages):
            start = page * plots_per_page
            end = start + plots_per_page
            current_columns = num_features[start:end]

            nrows = len(current_columns)
            fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(8, nrows * 3))

            if nrows == 1:
                axes = [axes]

            for i, feature in enumerate(current_columns):
                sns.boxplot(data=df, x=target_column, y=feature, ax=axes[i])
                axes[i].set_title(f'{feature} vs {target_column}')
                axes[i].set_xlabel(target_column)
                axes[i].set_ylabel(feature)

            plt.suptitle(f"Pairwise Relationships - Page {page + 1} / {total_pages}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
    
    def visualize_pairwise_relationships_categorical(self, target_column, plots_per_page=4):
        df = pd.read_csv(self.data_path)

        cat_features = df.select_dtypes(include=['category']).columns.tolist()

        if target_column not in df.columns:
            raise Exception(f"Target column '{target_column}' not found in the dataset.")

        if target_column in cat_features:
            cat_features.remove(target_column)

        if len(cat_features) < 1:
            raise Exception("Not enough categorical features to visualize pairwise relationships.")

        total = len(cat_features)
        total_pages = math.ceil(total / plots_per_page)

        for page in range(total_pages):
            start = page * plots_per_page
            end = start + plots_per_page
            current_columns = cat_features[start:end]

            nrows = len(current_columns)
            fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(8, nrows * 3))

            if nrows == 1:
                axes = [axes]

            for i, feature in enumerate(current_columns):
                sns.countplot(data=df, x=feature, hue=target_column, ax=axes[i])
                axes[i].set_title(f'{feature} vs {target_column}')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Count')

            plt.suptitle(f"Pairwise Relationships - Page {page + 1} / {total_pages}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

    def visualize_pairplot(self):
        df = pd.read_csv(self.data_path)
        number_columns = df.select_dtypes(include=[np.number])
        sns.pairplot(number_columns, diag_kind='kde', markers='o', plot_kws={'alpha': 0.5})
        plt.suptitle('Pairplot of Numerical Features', y=1.02, fontsize=16)
        plt.show()
        