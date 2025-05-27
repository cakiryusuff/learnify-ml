from config.config_paths import *
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer

class AutoMLPipeline:
    def __init__(self, target_column):
        self.target_column = target_column
        
    def run_pipeline(self):
        preprocessor = DataPreprocessor(self.target_column, DATA_PREPROCESSING_INPUT, DATA_PREPROCESSING_OUTPUT)
        preprocessor.run_preprocessing()

        trainer = ModelTrainer(self.target_column, DATA_PREPROCESSING_OUTPUT, MODEL_SAVE_PATH)
        trainer.run_training()