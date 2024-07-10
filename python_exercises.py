import pandas as pd

pd.set_option('display.max_columns', 100)
#TODO
# Ćwiczenie stworzyć kolumnę, która będzie zawierała true jeśli w nazwie kraju będzie "U" - zastosuj funkcję apply
# Cwiczenie poczytać o funkji apply i funkcji map

input_path = 'C:\\Users\\Ania\\Desktop\\ClimateChange\\processed_dataset.csv'
df = pd.read_csv(input_path)
print(df)

# letter_u_in_country =