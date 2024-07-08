import logging as logger
import pandas as pd
from load import LoadData, CheckData
import glob
import os
import numpy as np


#SETTINGS
data_path = 'C:\\Users\Ania\\Desktop\\ClimateChange_Natan\\'
date_list = ['dt']
merge_columns = ['Country', 'dt']
pd.set_option('display.max_columns', 100)

continents=['Asia', 'Africa', 'North America','South America', 'Europe', 'Australia','Antartica' ]
countries = {'Congo (Democratic Republic Of The)': 'Congo',
                     'Denmark (Europe)': 'Denmark',
                     'Falkland Islands (Islas Malvinas)': 'Falkland Islands',
                     'France (Europe)': 'France',
                     'Guinea Bissau': 'Guinea',
                     'Netherlands (Europe)': 'Netherlands',
                     'United Kingdom (Europe)': 'United Kingdom'
                    }

loader = LoadData()
checker = CheckData(
                    continents=continents,
                    countries=countries)

files_to_load = glob.glob(f'{data_path}*.csv')

dataset_list = list()
dataset_merged = pd.DataFrame()

for file in files_to_load:
    suffix = os.path.basename(file).replace('GlobalLandTemperatures', '').replace('.csv', '')
    dataset = loader.load_data(file_path=file)
    dataset = loader.parse_dates(dataset,
                          date_list)
    dataset_merged = loader.merge_data(dataset_merged,
                                       dataset,
                                       merge_columns, suffix=suffix)

dataset = checker.if_continents_in_countries_column(df=dataset_merged)
status = checker.check_nans(df=dataset_merged)
dataset_merged = checker.correct_country_names(df=dataset_merged)


# for i in dataset_merged['Country'].unique():
#     print(i)

#TODO
# Ćwiczenie stowrzyć kolumnę, która będzie zawierała true jeśli w nazwie kraju będzie "U" - zastosuj funkcję apply
# Cwiczenie poczytać o funkji apply i funkcji map

