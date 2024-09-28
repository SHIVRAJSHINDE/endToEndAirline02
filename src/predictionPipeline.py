import sys
import pandas as pd
from src.components.st01preProcessingTrainingFunctions import streamlineData
from src.components.st03preProcessingPredtionFunction import predctionFunctions


class CustomData():
        def __init__(self):
            self.streamline_data = streamlineData()
            self.predction_Functions = predctionFunctions()


        def receiveDataFromWeb(self,Airline:str,Date_of_Journey:str,Source:str,Destination:str,
                     Dep_Time:str,Arrival_Time:str,Duration:str,Total_Stops:str):
            
            inputDict = {
                "Airline": [Airline],
                "Date_of_Journey": [Date_of_Journey],
                "Source": [Source],
                "Destination": [Destination],
                "Dep_Time": [Dep_Time],
                "Arrival_Time": [Arrival_Time],
                "Duration": [Duration],
                "Total_Stops": [Total_Stops]

            }

            df = pd.DataFrame(inputDict)
            print(df)

            df = self.predction_Functions.extract_Deptdate_time(df)   
            df = self.predction_Functions.extract_Arrdate_time(df)
            df = self.streamline_data.extract_date_components(df)
            df = self.predction_Functions.calculateDuration(df)
            
            df = self.streamline_data.extract_Dept_Hrs_Minutes(df,'Dep_Time')
            df = self.streamline_data.extract_Arr_Hrs_Minutes(df,'Arrival_Time')
            columns_to_drop = ["Arrival_Time","Arrival_Time","Dep_Time","Date_of_Journey","Duration"]
            df = self.streamline_data.drop_unnecessary_columns(df,columns_to_drop)
            df = self.predction_Functions.restructure_columns_predictionPipeline(df)
            print(df.T)
            df = self.predction_Functions.preprocesing(df)
            ouput = self.predction_Functions.predict(df)
            print(df.T)
            return ouput
