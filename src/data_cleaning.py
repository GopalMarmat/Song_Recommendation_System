#Importing Library
import pandas as pd
import os


#Reading Traning file
directory='C:/Users/HP/Documents/AIML_Project/Song_Recommendation_System/Data'
File_name='Song_Recommendion_Train.csv'
Train_csv_path=os.path.join(directory,File_name)
data = pd.read_csv(Train_csv_path, encoding='ISO-8859-1')


# Remove transactions where the 'year' value is zero
data = data[data['year'] > 0]

#remove duplicate
data.drop_duplicates(inplace=True)


#Store Clean data csv file for model building.
Clean_File_Name='modified_data.csv'
Clean_csv_path=os.path.join(directory,Clean_File_Name)
data.to_csv(Clean_csv_path, index=False)