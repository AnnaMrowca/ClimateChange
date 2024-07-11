import pandas as pd
from load import LoadData, CheckData, Model, Visual
import glob
import os

#SETTINGS
data_path = 'C:\\Users\Ania\\Desktop\\ClimateChange\\'
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

modeling_columns = ['dt', 'Country', 'AverageTemperatureByState', 'StateByState']

loader = LoadData()
checker = CheckData(continents=continents,
                    countries=countries)
modeler = Model(modeling_columns=modeling_columns)
visualizer = Visual()

files_to_load = glob.glob(f'{data_path}*.csv')

dataset_list = list()
dataset_merged = pd.DataFrame()

for file in files_to_load:
    suffix = os.path.basename(file).replace('GlobalLandTemperatures', '').replace('.csv', '')
    dataset = loader.load_data(file_path=file)
    dataset = loader.parse_dates(dataset,
                          date_list)
    #check size of file
# print(dataset.memory_usage(index=True).sum()/1073741824)
    dataset_merged = loader.merge_data(dataset_merged,
                                       dataset,
                                       merge_columns, suffix=suffix) #Natan, czy to nie powinno być poza forem?

dataset = checker.if_continents_in_countries_column(df=dataset_merged)
status = checker.check_nans(df=dataset_merged)
dataset_merged = checker.correct_country_names(df=dataset_merged)

visualizer.create_visuals(time_series=dataset_merged['dt'],
                          temperature_series = dataset_merged['AverageTemperatureByState'])

model_dataset = modeler.get_modeling_columns(df=dataset_merged,
                                              modeling_columns=modeling_columns)
model_dataset = modeler.drop_nans(df=model_dataset,
                                  modeling_columns=modeling_columns)

test, train = modeler.test_train_split(df=dataset_merged)




# output_path = 'C:\\Users\\Ania\\Desktop\\ClimateChange_Natan\\processed_dataset.csv' #trzeba zaposać w innym folderze
# dataset_merged.to_csv(output_path, index=False)




# for i in dataset_merged['Country'].unique():
#     print(i)

#TODO
# Ćwiczenie stowrzyć kolumnę, która będzie zawierała true jeśli w nazwie kraju będzie "U" - zastosuj funkcję apply
# Cwiczenie poczytać o funkji apply i funkcji map

