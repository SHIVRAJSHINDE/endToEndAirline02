
import pandas as pd
from src.exp.utils import load_objects


class predctionFunctions():

    def extract_Deptdate_time(self,df):
        # Split the specified column into 'DeptDATE' and 'DeptTIME'
        df[['Date_of_Journey', 'Dep_Time']] = df['Date_of_Journey'].str.split('T', expand=True)
        df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey']).dt.strftime('%d/%m/%Y')

        return df
    def extract_Arrdate_time(self,df):
        # Split the specified column into 'DeptDATE' and 'DeptTIME'
        df[['Arrival_Time']] = df['Arrival_Time'].str.split("T")[0][1]        
        return df

    def calculateDuration(self,df):
        # Calculate Duration
        df['Arrival_Time'] = pd.to_datetime(df['Arrival_Time'])
        df['Dep_Time'] = pd.to_datetime(df['Dep_Time'])
        # Convert time to datetime format
        df['Duration'] = df['Arrival_Time'] - df['Dep_Time']
        # Calculate total minutes
        df['hoursMinutes'] = df['Duration'].dt.total_seconds() / 60

        return df
    
    def restructure_columns_predictionPipeline(self,df):
        df = df[['Airline', 'Source', 'Destination', 'Total_Stops', 'Day','Month', 'Year',
                        'Dept_Hour', 'Dept_Minute', 'Arr_Hour', 'Arr_Minute','hoursMinutes']]
        
        return df


    def preprocesing(self,df):
        preprocessorPath = "artifacts/preprocessor.pkl"
        transformation = load_objects(file_path=preprocessorPath)
        dataScaled = transformation.transform(df)
        return dataScaled


    def predict(self,dataScaled):
        modelPath = "artifacts/best_model.pkl"
        model = load_objects(file_path=modelPath)
        pred = model.predict(dataScaled)
        return pred
