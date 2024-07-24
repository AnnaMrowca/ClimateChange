import pandas as pd
import matplotlib.pyplot as plt
from load import LoadData, CheckData, Model, Visual
import glob
import os

# SETTINGS
data_path = "C:\\Users\Ania\\Desktop\\ClimateChange\\"
date_list = ["dt"]
merge_columns = ["Country", "dt"]
pd.set_option("display.max_columns", 100)

continents = [
    "Asia",
    "Africa",
    "North America",
    "South America",
    "Europe",
    "Australia",
    "Antartica",
]
countries = {
    "Congo (Democratic Republic Of The)": "Congo",
    "Denmark (Europe)": "Denmark",
    "Falkland Islands (Islas Malvinas)": "Falkland Islands",
    "France (Europe)": "France",
    "Guinea Bissau": "Guinea",
    "Netherlands (Europe)": "Netherlands",
    "United Kingdom (Europe)": "United Kingdom",
}

modeling_columns = ["dt", "Country", "AverageTemperatureByCountry"]
modeling_country = [
    "Poland"
]  # dt, country = Poland, avegage temperature, Arima, Sarimax, Prophet (sezonowość)
three_sigma_col = ["AverageTemperatureByCountry"]
adf_column = ["AverageTemperatureByCountry"]

loader = LoadData()
checker = CheckData(continents=continents, countries=countries)
modeler = Model()

visualizer = Visual()

files_to_load = glob.glob(f"{data_path}*.csv")

dataset_list = list()
dataset_merged = pd.DataFrame()

for file in files_to_load:
    suffix = (
        os.path.basename(file).replace("GlobalLandTemperatures", "").replace(".csv", "")
    )
    dataset = loader.load_data(file_path=file)
    dataset = loader.parse_dates(dataset, date_list)

    dataset_merged = loader.merge_data(
        dataset_merged, dataset, merge_columns, suffix=suffix
    )

dataset = checker.if_continents_in_countries_column(df=dataset_merged)
status = checker.check_nans(df=dataset_merged)
dataset_merged = checker.correct_country_names(df=dataset_merged)

model_dataset = modeler.get_modeling_columns(
    df=dataset_merged, modeling_columns=modeling_columns
)
model_dataset = modeler.drop_nans(df=model_dataset, modeling_columns=modeling_columns)

model_dataset = modeler.select_country(
    df=model_dataset, modeling_country=modeling_country
)

model_dataset = modeler.remove_3sigma_outliers(
    df=model_dataset, three_sigma_col=three_sigma_col
)

fig, ax = visualizer.create_visuals_scatterplot(
    df=model_dataset.loc[model_dataset["dt"] >= pd.to_datetime("01-01-2009")],
    time_series="dt",
    temperature_series="AverageTemperatureByCountry",
)

model_dataset_trend_line = visualizer.add_trending_line(
    df=model_dataset.loc[model_dataset["dt"] >= pd.to_datetime("01-01-2009")],
    time_series="dt",
    temperature_series="AverageTemperatureByCountry",
    fig=fig,
    ax=ax,
)
plt.show()

train, test = modeler.test_train_split(df=model_dataset)

adf_testing = modeler.testing_adf(df=train, adf_column=adf_column)

model = modeler.get_auto_arima(
    train_data=train,
    temperature_series="AverageTemperatureByCountry",
    seasonal=True,
    m=6,
)

forecast = modeler.get_forecast(
    model=model,
    initial_date=test["dt"].min(),
    n_periods=len(test) + 12,
)

arima_visual = visualizer.create_forecast_visual(
    train_data=train.loc[train["dt"] >= pd.to_datetime("01-01-1953")],
    test_data=test,
    forecast=forecast,
    temperature_series="AverageTemperatureByCountry",
    time_series="dt",
)
plt.show()

prophet_model = modeler.get_prophet_model(train_data=train)

test_forecast, future_forecast = modeler.get_prophet_forecast(
    model=prophet_model, test_data=test, n_periods=len(test) + 12
)

prophet_visual = visualizer.create_forecast_visual(
    train_data=train.loc[train["dt"] >= pd.to_datetime("01-01-1953")],
    test_data=test,
    forecast=future_forecast,
    temperature_series="AverageTemperatureByCountry",
    time_series="dt",
)
plt.show()
