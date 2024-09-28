import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys
from src.exp.utils import load_objects
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler



class streamlineData():
    def __init__(self):
        self.df = None
        self.final_df = None
        self.airline_quantiles = {
            'IndiGo': [0.25, 0.75],
            'Air India': [0.27, 0.75],
            'Jet Airways': [0.27, 0.75],
            'SpiceJet': [0.10, 0.60],
            'Multiple carriers': [0.20, 0.80],
            'GoAir': [0.20, 0.75],
            'Vistara': [0.20, 0.75],
            'Air Asia': [0.25, 0.75],
            'Vistara Premium economy': [0.25, 0.75],
            'Jet Airways Business': [0.25, 0.75],
            'Multiple carriers Premium economy': [0.20, 0.75],
            'Trujet': [0, 0]
        }

    def read_data(self,filePath):
        print(filePath)
        """Read the Excel file into a DataFrame."""
        self.df = pd.read_excel(filePath)
        return self.df

    def drop_missing_rows(self,df):
        """Identify and drop rows with missing values in the 'Route' column."""
        # Find indices of rows where 'Route' is null
        route_missing_rows = df[df['Route'].isnull()].index
        # Drop those rows from the DataFrame
        df.drop(route_missing_rows, inplace=True)
        return df
    
    def drop_duplicates(self,df):
        df = df.drop_duplicates()
        return df

    def basiCleanUp(self,df):
        
        df['Destination'].replace(to_replace="New Delhi",value="Delhi",inplace=True)
        return df

    def remove_outliers_by_airline(self,df):
        """
        Remove outliers from the DataFrame based on airline-specific quantile ranges.
        
        Returns:
        pd.DataFrame: DataFrame with outliers removed.
        """

        final_df = pd.DataFrame(columns=list(df.columns))

        # Iterate through each airline and apply the outlier removal
        for airline, quantiles in self.airline_quantiles.items():
            airDataSet = self.df[self.df['Airline'] == airline]

            if airDataSet.shape[0] > 5:
                # Calculate Q1, Q3, and IQR
                q1 = airDataSet['Price'].quantile(quantiles[0])
                q3 = airDataSet['Price'].quantile(quantiles[1])
                IQR = q3 - q1
                
                # Determine the lower and upper limits
                lowerLimit = q1 - 1.5 * IQR
                upperLimit = q3 + 1.5 * IQR
                
                # Filter the dataset to remove outliers
                airDataSet = airDataSet[(airDataSet['Price'] >= lowerLimit) & (airDataSet['Price'] <= upperLimit)]
            
            # Append the filtered dataset to the final DataFrame
            final_df = pd.concat([final_df, airDataSet], axis=0)
            self.final_df = final_df 
            #print(final_df)
        
        return self.final_df


    def extract_date_components(self,df):
        # Ensure 'Date_of_Journey' column exists
        if 'Date_of_Journey' not in df.columns:
            raise ValueError("The DataFrame must contain a 'Date_of_Journey' column.")
        
        # Convert 'Date_of_Journey' to datetime and extract components
        df['Day'] = pd.to_datetime(df['Date_of_Journey'], format="%d/%m/%Y").dt.day
        df['Month'] = pd.to_datetime(df['Date_of_Journey'], format="%d/%m/%Y").dt.month
        df['Year'] = pd.to_datetime(df['Date_of_Journey'], format="%d/%m/%Y").dt.year

        return df
    
    def extract_Dept_Hrs_Minutes(self, df,col_Name):
        df['Dept_Hour']=pd.to_datetime(df[col_Name]).dt.hour
        df['Dept_Minute']=pd.to_datetime(df[col_Name]).dt.minute

        return df

    def extract_Arr_Hrs_Minutes(self, df,col_Name):
        df['Arr_Hour']=pd.to_datetime(df[col_Name]).dt.hour
        df['Arr_Minute']=pd.to_datetime(df[col_Name]).dt.minute

        return df

    def calculate_hours_minutes(self,df):
        """Calculate total minutes from the 'Duration' column and store in 'hoursMinutes'."""
        df["hoursMinutes"] = 0  # Initialize the new column

        for i in df.index:
            duration = df.loc[i, 'Duration']
            
            if " " in duration:
                column1, column2 = duration.split(" ")
            else:
                column1, column2 = duration, ""

            # Function to convert duration string to minutes
            def convert_to_minutes(column):
                if "h" in column:
                    return int(column.replace("h", "")) * 60
                elif "m" in column:
                    return int(column.replace("m", ""))
                return 0
            
            # Calculate total minutes
            total_minutes = convert_to_minutes(column1) + convert_to_minutes(column2)
            df.loc[i, 'hoursMinutes'] = total_minutes

        return df

    def drop_unnecessary_columns(self,df, columns_to_drop):
        """Drop unnecessary columns from the DataFrame."""
        
        df = df.drop(columns=columns_to_drop, axis=1)
        return df

    def restructure_columns(self,df):
        df = df[['Airline', 'Source', 'Destination', 'Total_Stops', 'Day','Month', 'Year',
                         'Dept_Hour', 'Dept_Minute', 'Arr_Hour', 'Arr_Minute','hoursMinutes','Price']]
        
        return df

    def y_column_and_X_columns(self,df):
        y = df['Price']
        X = df.drop('Price', axis=1)
        
        return y, X
    
    def train_test_split(self,y,X):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def methodPreprocessing(self):
        trf1 = ColumnTransformer([
            ('OneHot',OneHotEncoder(drop='first',handle_unknown='ignore'),[0,1,2])],remainder='passthrough')
        
        trf2 = ColumnTransformer([
            ('Ordinal',OrdinalEncoder(categories=[['non-stop', '1 stop', '2 stops', '3 stops', '4 stops']]),[19])]
            ,remainder='passthrough')
        trf3 = ColumnTransformer([
            ('scale', StandardScaler(), slice(0, 28))
        ])
        pipe = make_pipeline(trf1,trf2,trf3)

        # Save the pipeline as preprocessor.pkl
        #os.makedirs('artifacts', exist_ok=True)
        #with open('artifacts/preprocessor.pkl', 'wb') as f:
        #    pickle.dump(pipe, f)
        #print("Preprocessing pipeline saved as preprocessor.pkl")
        
        return pipe



