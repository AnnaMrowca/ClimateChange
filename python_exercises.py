import pandas as pd

pd.set_option("display.max_columns", 100)
# TODO
# Ćwiczenie stworzyć kolumnę, która będzie zawierała true jeśli w nazwie kraju będzie "U" - zastosuj funkcję apply
# Cwiczenie poczytać o funkji apply i funkcji map

input_path = (
    "C:\\Users\\Ania\\Desktop\\ClimateChange\\GlobalLandTemperaturesByCountry.csv"
)
df = pd.read_csv(input_path)

letter_u = df["Country"].apply(lambda country: "u" in country.lower())
countries_with_u = df[letter_u]

contains_93 = df["AverageTemperature"].apply(
    lambda temperature: "93" in str(temperature)
)
temperatures_with_93 = df[contains_93]
print(temperatures_with_93)
