import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass, field

from src.components.st01preProcessingTrainingFunctions import streamlineData
from src.exp.utils import save_object
from src.components.st02trainingFunctions import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        self.filePath = "D:\\MLProjects\\ZDatasets\\AirlineData\\r\\Combined.xlsx"
        self.streamline_data = streamlineData()
        self.ModelTrainer = ModelTrainer()

    def training_pipeline_method(self):
        # Read the data from the file
        df = self.streamline_data.read_data(self.filePath)
        df = self.streamline_data.drop_missing_rows(df)
        
        df = self.streamline_data.drop_duplicates(df)
        df = self.streamline_data.basiCleanUp(df)
        print(df)
        print("--------------------------------")

        # Remove outliers by airline
        df = self.streamline_data.remove_outliers_by_airline(df)
        # Extract date components
        df = self.streamline_data.extract_date_components(df)
        df = self.streamline_data.extract_Dept_Hrs_Minutes(df,'Dep_Time')
        df = self.streamline_data.extract_Arr_Hrs_Minutes(df,'Arrival_Time')
        df = self.streamline_data.calculate_hours_minutes(df)
        columns_to_drop = ["Arrival_Time","Arrival_Time","Dep_Time","Date_of_Journey","Route","Duration","Additional_Info"]
        df = self.streamline_data.drop_unnecessary_columns(df,columns_to_drop)
        df = self.streamline_data.restructure_columns(df)
        y,X = self.streamline_data.y_column_and_X_columns(df)
        X_train, X_test, y_train, y_test = self.streamline_data.train_test_split(y,X)
        pipe = self.streamline_data.methodPreprocessing()
        # print(X_train['Airline'].value_counts())
        X_train = pipe.fit_transform(X_train)
        X_test = pipe.transform(X_test)

        save_object(file_path='artifacts/preprocessor.pkl',obj=pipe)
        self.ModelTrainer.train_and_save_best_model(X_train, y_train, X_test, y_test)

        return X_train
    

# Example usage
if __name__ == "__main__":
    pipeline = TrainingPipeline()
    df = pipeline.training_pipeline_method()
    print(df.T)
