from typing import List, Tuple
import pandas as pd
import numpy as np
from typing import Literal
from learnify_ml.utils.common_functions import load_data, save_data
from learnify_ml.src.custom_exception import CustomException
from learnify_ml.src.logger import get_logger
from learnify_ml.config.config_paths import *

logger = get_logger(__name__)

class DataPreprocessor():
    """Initialize the DataPreprocessor with target column and paths.
    
    This class handles data preprocessing tasks such as handling missing values, scaling, encoding, feature selection, and balancing the dataset.
    It includes methods for preprocessing the data, handling missing values, removing outliers, scaling numeric features, label encoding, feature selection, variance inflation factor (VIF) calculation, skewness treatment, and balancing the dataset using SMOTE.
    It also provides a method to run the complete preprocessing pipeline.
    
    Parameters
    ----------
    
    target_column (str) : The name of the target column in the dataset
    data_path (str): The path to the input dataset.
    data_output_path (str): The path where the preprocessed dataset will be saved.
    impute_strategy (str): The strategy to use for imputation (e.g., 'mean', 'median', 'most_frequent').
    impute_strategy_remove (str): The strategy to use for imputation (e.g., 'column', 'row', None).
    apply_smote (bool): Whether to apply SMOTE (Synthetic Minority Over-sampling Technique) or not.
    apply_scale (bool): Whether to apply scaling or not.
    apply_outlier (bool): Whether to apply outlier detection or not.
    apply_vif (bool): Whether to apply VIF (Variance Inflation Factor) or not.
    apply_skewness (bool): Whether to apply skewness detection or not.
    """
    
    def __init__(self, 
                target_column: str = ...,
                data_path: str | None = DATA_PREPROCESSING_INPUT,
                data_output_path: str | None = DATA_PREPROCESSING_OUTPUT,
                impute_strategy: Literal["mean", "median", "most_frequent"] = "mean",
                impute_strategy_remove: Literal["column", "row", None] = None,
                apply_scale: bool = False,
                apply_skewness: bool = False,
                encode_target_column: bool = False,
                apply_outlier: bool = False,
                apply_vif: bool = False,
                apply_smote: bool = True):
        
        self.target_column = target_column
        self.data_path = data_path
        self.data_output_path = data_output_path
        self.impute_strategy = impute_strategy
        self.impute_strategy_remove = impute_strategy_remove
        self.apply_scale = apply_scale
        self.apply_skewness = apply_skewness
        self.encode_target_column = encode_target_column
        self.apply_outlier = apply_outlier
        self.apply_vif = apply_vif
        self.apply_smote = apply_smote

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # TODO: Add more preprocessing steps as needed (e.g., imputation, scaling, etc.)
            # TODO: Make this function more dynamic to handle different datasets
            logger.info("Starting data preprocessing")
            
            df = df.copy()
            
            df = df.dropna(axis=1, how='all')
            
            df = df.drop_duplicates()
        
            # TODO: Make this feature engineering stuff more dynamic, it can be done using AI agents
            # df["Title"] = df["Name"].apply(lambda x: x.split(",")[1].split(".")[0])
            # df["Age"] = df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean"))
            # df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
            # df.drop(["Name", "Ticket", "Cabin", "PassengerId", "SibSp", "Parch"], axis=1, inplace=True)
            
            return df
        
        except Exception as e:
            logger.error(f"Error in preprocessing data: {e}")
            raise CustomException(e, "Error in preprocessing data")
        
    def split_object_columns(self, df: pd.DataFrame, max_unique_thresh:int = 30, min_avg_len_thresh: int=20) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Split object columns into categorical and text columns based on unique values and average length.
        Parameters:
            - df (pd.DataFrame): The DataFrame containing the data to be split.
            - max_unique_thresh (int): The maximum number of unique values for a column to be considered categorical.
            - min_avg_len_thresh (int): The minimum average length for a column to be considered text.
        Returns:
            - tuple: A tuple containing the list of categorical columns and the list of text columns.
        """
        try:
            logger.info("Splitting object columns into categorical and text columns")
            df = df.copy()
            
            categorical_cols = []
            text_cols = []

            for col in df.select_dtypes(include='object').columns:
                num_unique = df[col].nunique()
                avg_len = df[col].astype(str).apply(len).mean()

                if num_unique <= max_unique_thresh and avg_len <= min_avg_len_thresh:
                    categorical_cols.append(col)
                else:
                    text_cols.append(col)

            df = df.drop(columns=text_cols)
            
            logger.info(f"- Number of categorical columns: {len(categorical_cols)}, categorical columns: {categorical_cols}")
            logger.info(f"- Number of text columns: {len(text_cols)}, text columns: {text_cols}")
            
            return df, categorical_cols, text_cols
        
        except Exception as e:
            logger.error(f"Error in splitting object columns: {e}")
            raise CustomException(e, "Error in splitting object columns")
   
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data with potential missing values.
        
        Returns:
        pd.DataFrame: The DataFrame with missing values handled.
        """
        try:
            logger.info("Handling missing values in the DataFrame")
            
            df = df.copy()
            
            if self.impute_strategy_remove is None:
                logger.info("- Imputing missing values based on the specified strategy")
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        if self.impute_strategy == 'mean':
                            df[col] = df[col].fillna(df[col].mean())
                        elif self.impute_strategy == 'median':
                            df[col] = df[col].fillna(df[col].median())
                        elif self.impute_strategy == 'most_frequent':
                            df[col] = df[col].fillna(df[col].mode()[0])
                    else:
                        df[col] = df[col].fillna('Unknown')
                return df
            else:
                logger.info("- Removing rows or columns with missing values based on the specified strategy")
                if self.impute_strategy_remove == "column":
                    df = df.dropna(axis=1, how='any')
                elif self.impute_strategy_remove == "row":
                    df = df.dropna(axis=0, how='any')
                else:
                    raise ValueError("Invalid impute_strategy_remove value. Use 'column' or 'row'.")
                return df
        except Exception as e:
            logger.error(f"Error in handling missing values: {e}")
            raise CustomException(e, "Error in handling missing values")
    
    def remove_outliers(
        self,
        df: pd.DataFrame,
        method: Literal["zscore", "iqr"] = "iqr",
        threshold: float = 3.0,
        columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Remove outliers from numeric columns using Z-Score or IQR method.

        Parameters:
        - df (pd.DataFrame): Input DataFrame.
        - method (str): "zscore" or "iqr".
        - threshold (float): Threshold value. For z-score it's standard deviations (default=3),
                            for IQR it's multiplier (default=1.5 is recommended).
        - columns (List[str], optional): Specific columns to apply; if None, applies to all numeric.

        Returns:
        - pd.DataFrame: DataFrame with outliers removed.
        """
        try:
            if not self.apply_outlier:
                logger.info("Skipping removal of outliers as apply_outlier is set to False")
                return df
            
            df = df.copy()

            logger.info("Removing outliers from numeric columns using Z-Score or IQR method")
            df_cleaned = df.copy()
            numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()
            
            if columns is None:
                columns = numeric_cols
            else:
                columns = [col for col in columns if col in numeric_cols]

            if method == "zscore":
                from scipy.stats import zscore
                z_scores = np.abs(zscore(df_cleaned[columns], nan_policy='omit'))
                mask = (z_scores < threshold).all(axis=1)
                df_cleaned = df_cleaned[mask]
            elif method == "iqr":
                for col in columns:
                    Q1 = df_cleaned[col].quantile(0.25)
                    Q3 = df_cleaned[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
            else:
                raise ValueError("method must be either 'zscore' or 'iqr'")
            
            return df_cleaned.reset_index(drop=True)
        
        except Exception as e:
            logger.error(f"Error in removing outliers: {e}")
            raise CustomException(e, "Error in removing outliers")
    
    def scale_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numeric features in the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data to be scaled.
        
        Returns:
        pd.DataFrame: The DataFrame with scaled numeric features.
        """
        try:
            if not self.apply_scale:
                logger.info("Skipping scaling of numeric features as scale_numeric is set to False")
                return df
            
            df = df.copy()
            
            logger.info("Scaling numeric features in the DataFrame")
            from sklearn.preprocessing import StandardScaler
            
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            numeric_cols = [col for col in numeric_cols if col != self.target_column]
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            
            return df
        except Exception as e:
            logger.error(f"Error in scaling numeric features: {e}")
            raise CustomException(e, "Error in scaling numeric features")
    
    def label_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label encode categorical columns in the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data to be encoded.
        
        Returns:
        pd.DataFrame: The DataFrame with label encoded categorical columns.
        """
        try:
            logger.info("Starting label encoding of categorical columns")
            
            df = df.copy()
            
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = df[col].astype('category').cat.codes
                
            if self.encode_target_column and self.target_column in df.columns:
                logger.info(f"Label encoding target column: {self.target_column}")
                df[self.target_column] = df[self.target_column].astype('category').cat.codes
                
            return df
        except Exception as e:
            logger.error(f"Error in label encoding: {e}")
            raise CustomException(e, "Error in label encoding")

    def feature_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select features based on correlation with the target variable.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        
        Returns:
        pd.DataFrame: The DataFrame with selected features.
        """
        try:
            logger.info("Selecting features based on correlation with target variable")
            df = df.copy()
            corr_matrix = df.corr()
            target_corr = corr_matrix[self.target_column].abs().sort_values(ascending=False)
            selected_features = target_corr[target_corr > 0.1].index.tolist()
            return df[selected_features]
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            raise CustomException(e, "Error in feature selection")
    
    def variance_inflation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate and remove features with high variance inflation factor (VIF).
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        
        Returns:
        pd.DataFrame: The DataFrame with features having high VIF removed.
        """
        try:
            if not self.apply_vif:
                logger.info("Skipping variance inflation factor (VIF) calculation as apply_vif is set to False")
                return df
            
            df = df.copy()
            
            logger.info("Calculating variance inflation factor (VIF) for features")
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            
            vif_data = pd.DataFrame()
            vif_data["feature"] = df.columns
            vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
            high_vif_features = vif_data[vif_data["VIF"] > 10]["feature"].tolist()
            df.drop(columns=high_vif_features, inplace=True)
            
            logger.info(f"Features with high VIF removed: {high_vif_features}")
            return df
        except Exception as e:
            logger.error(f"Error in calculating variance inflation factor: {e}")
            raise CustomException(e, "Error in calculating variance inflation factor")
        
    def skewness_treatment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log transformation to skewed numerical features.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        
        Returns:
        pd.DataFrame: The DataFrame with skewed features transformed.
        """
        try:
            if not self.apply_skewness:
                logger.info("Skipping skewness treatment as apply_skewness is set to False")
                return df
            
            logger.info("Applying skewness treatment to numerical features")
            skewed_features = df.select_dtypes(include=['float64', 'int64']).apply(lambda x: x.skew()).sort_values(ascending=False)
            skewed_features = skewed_features[skewed_features > 2].index
            
            for feature in skewed_features:
                df[feature] = np.log1p(df[feature])
            
            logger.info(f"- Skewness applied to this numerical features: {skewed_features}")
            return df
        except Exception as e:
            logger.error(f"Error in skewness treatment: {e}")
            raise CustomException(e, "Error in skewness treatment")
        
    def balance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance the dataset using random oversampling.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        target_col (str): The name of the target column.
        
        Returns:
        pd.DataFrame: The balanced DataFrame.
        """
        try:
            
            if not self.apply_smote:
                logger.info("Skipping balancing data as apply_smote is set to False")
                return df
            
            df = df.copy()
            
            logger.info("Balancing the dataset using SMOTE")
            from imblearn.over_sampling import SMOTE
            
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
            
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            return pd.concat([X_resampled, y_resampled], axis=1)
        except Exception as e:
            logger.error(f"Error in balancing data: {e}")
            raise CustomException(e, "Error in balancing data")

    def run_preprocessing(self) -> None:
        """
        Run the complete preprocessing pipeline.
        
        Returns:
        pd.DataFrame: The preprocessed DataFrame.
        """
        try:
            logger.info("------------------------------------------------------")
            logger.info("Running complete preprocessing pipeline")
            
            df = load_data(self.data_path)

            df = self.preprocess_data(df)

            df, _, _ = self.split_object_columns(df) #? categorical and text columns can be used later

            df = self.remove_outliers(df)

            df = self.handle_missing_values(df)

            df = self.skewness_treatment(df)

            df = self.scale_numeric_features(df)

            df = self.label_encode(df)

            df = self.feature_selection(df)

            df = self.variance_inflation(df)

            df = self.balance_data(df)
            
            save_data(df, self.data_output_path)
            logger.info("Preprocessing completed successfully")
        except Exception as e:
            logger.error(f"Error in running preprocessing pipeline: {e}")
            raise CustomException(e, "Error in running preprocessing pipeline")

if __name__ == "__main__":
    preprocessor = DataPreprocessor(target_column="Survived")
    preprocessed_data = preprocessor.run_preprocessing()
    print("Preprocessing completed. Preprocessed data saved to:", DATA_PREPROCESSING_OUTPUT)