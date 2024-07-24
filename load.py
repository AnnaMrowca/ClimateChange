import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging as logger

from collections import Counter
from statsmodels.tsa.stattools import adfuller

import pmdarima as pm
from prophet import Prophet


logger.basicConfig(level=logger.INFO)


class LoadData:
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Loads data from a CSV file located at the file_path into a DataFrame.

        Parameters:
        - file_path: ath to the CSV file.

        Returns:
        - df: DataFrame with data from the CSV file, or an empty DataFrame if the file is not csv
        """

        try:
            file = pd.read_csv(file_path)
            return file

        except ValueError as e:
            logger.warning(f"{e} error. File at {file_path} is not csv")
            return pd.DataFrame()

    def parse_dates(self, df: pd.DataFrame, date_list: list = []) -> pd.DataFrame:
        """
        Parses date columns in a DataFrame to datetime format based on the provided date_list.

        Parameters:
        - df: DataFrame - DataFrame containing columns to parse.
        - date_list: list of column names in df to parse as dates. Default is an empty list.

        Returns:
        - df: DataFrame with specified columns converted to datetime format.
        """

        for column in date_list:
            if column in df.columns:
                df[column] = pd.to_datetime(df[column])
        return df

    def merge_data(
        self, df_1: pd.DataFrame, df_2: pd.DataFrame, merge_columns: list, suffix: str
    ) -> pd.DataFrame:
        """
        Merges two DataFrames and returns the merged DataFrame.

        Parameters:
        - df_1: DataFrame - first DataFrame
        - df_2: DataFrame - second DataFrame
        - merge_columns: list of columns to merge on
        - suffix: str - suffix to append to column names from df_2 if not present in df_1

        Returns:
        - df: Merged DataFrame combining df_1 and df_2 based on merge_columns
        """

        for column in df_2.columns:
            if column not in merge_columns:
                df_2 = df_2.rename(columns={column: f"{column}{suffix}"})

        if df_1.empty:
            return df_2

        if df_2.empty:
            return df_1

        for item in merge_columns:
            if item not in df_2.columns:
                df_name = f"{df_2=}".split("=")[0]
                logger.warning(
                    f"{df_name} does not contain {item} and cannot be merged"
                )
                return df_1

        df_total = pd.merge(df_1, df_2, on=merge_columns, how="outer")
        return df_total


class CheckData:
    def __init__(self, continents, countries):
        self.continents = continents
        self.countries = countries

    def check_nans(self, df: pd.DataFrame) -> bool:
        """
        Checks for NaN values in the dataset columns.

        Parameters:
        - dataset: DataFrame with data to be checked for NaN values.

        Returns:
        - bool: False if there are NaN values in columns, True otherwise.
        """

        columns_with_nans = []
        for column in df.columns:
            if df[column].isna().any():
                columns_with_nans.append(column)

        if columns_with_nans:
            print(f"Check NaNs - False. Columns with NaNs: {columns_with_nans}")
            return False
        else:
            print("Check NaNs - True: dataset does not contain any NaN values")
            return True

    def if_continents_in_countries_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a 'Continent_in_Country' column with True/False values based on whether the 'Country'
        column contains a continent (True), and logs percentage of True entries in the dataset.

        Parameters:
        - df: the DataFrame with the 'Country' column to check.

        Returns:
        - df: updated DataFrame with the 'Continent_in_Country' column
        """
        df["Continent_in_Country"] = df["Country"].isin(self.continents)
        count = df["Continent_in_Country"].sum()
        error_percent = round((count / len(df)) * 100, 2)
        logger.warning(
            f"{count} entries have continent in country. It is {error_percent} % of the entire dataset "
        )
        return df

    def correct_data_in_columns(
        self, df: pd.DataFrame, corrections: dict, column: str
    ) -> pd.DataFrame:
        """
        Generic function to correct data in specific column using correction dict

        Parameters:
        - df: DataFrame with data to be corrected.
        - corrections: dictionary mapping incorrect and correct values.
        - column: column name where corrections should be applied.

        Returns:
        - df: DataFrame with corrected data in specified column.
        """

        col_changes_count = Counter()

        for original_value, corrected_value in corrections.items():
            mask = df[column] == original_value
            count = mask.sum()
            if count > 0:
                col_changes_count[original_value] += count

        logger.warning(f"Columns changed: {dict(col_changes_count)}")
        df[column] = df[column].replace(corrections)
        return df

    def correct_country_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Corrects country names in a DataFrame using the `countries` dictionary

        Parameters:
        - df: DataFrame with data to be corrected

        Returns:
        - df: DataFrame with corrected data in 'Country' column
        """
        df = self.correct_data_in_columns(df, self.countries, "Country")
        return df


class Model:

    def get_modeling_columns(self, df: pd.DataFrame, modeling_columns) -> pd.DataFrame:
        """
        Extracts specified modeling columns from a df and logs the percentage of missing values

        Parameters:
        - df: DataFrame to extract the modeling columns from
        - modeling_columns: list of columns to filter from the df

        Returns:
        - modeling_df: DataFrame containing the specified modeling columns
        """

        modeling_df = df.loc[:, modeling_columns]
        nans = modeling_df[modeling_columns].isna().sum().sum()
        nans_percent = round(nans / len(modeling_df) * 100, 2)
        if nans_percent < 10:
            logger.info(
                f"LOW number of not existent values in modeling column {modeling_columns}: {nans_percent}%"
            )
        else:
            logger.warning(
                (
                    f"HIGH number of non existent values in modeling column {modeling_columns}: {nans_percent}%"
                )
            )
        return modeling_df

    def drop_nans(self, df: pd.DataFrame, modeling_columns) -> pd.DataFrame:
        """
        Drops rows with NaN values in modeling columns

        Parameters:
        - df: DataFrame from which to drop rows with NaN values
        - modeling_columns: list of column names to check for NaN values

        Returns:
        - df: DataFrame cleaned of NaN values in modeling columns
        """
        df = df.dropna(subset=modeling_columns)
        return df

    def test_train_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Splits the DataFrame into training and testing sets using an 80-20 split

        Parameters:
        - df: DataFrame to be split into training and testing sets

        Returns:
        - train: DataFrame, 80% of the original df for training
        - test: DataFrame, 20% of the original df for testing
        """
        df.reset_index(inplace=True)
        split_index = round((df.index.max() * 0.8))
        train = df.loc[df.index <= split_index]
        test = df.loc[df.index > split_index]

        return train, test

    def select_country(self, df: pd.DataFrame, modeling_country) -> pd.DataFrame:
        """
        Selects rows where the 'Country' column matches the specified country.

        Parameters:
        - df: DataFrame to filter
        - country: Country name to filter by

        Returns:
        - df: Filtered DataFrame containing only rows where 'Country' is the specified country
        """
        df = df.loc[df["Country"].isin(modeling_country)]
        return df

    def remove_3sigma_outliers(
        self, df: pd.DataFrame, three_sigma_col: list
    ) -> pd.DataFrame:
        """
        Removes outliers from the specified columns using the 3-sigma rule.

        Parameters:
        - df: DataFrame to remove outliers from
        - three_sigma_col: list of column to apply the 3-sigma rule

        Returns:
        - df_cleaned: DataFrame with outliers removed based on the 3-sigma rule
        """

        df_cleaned = df.copy()

        for col in three_sigma_col:
            mean = df_cleaned[col].mean()
            sd = df_cleaned[col].std()

            lower_bound = mean - (3 * sd)
            upper_bound = mean + (3 * sd)

            outliers_mask = (df_cleaned[col] < lower_bound) | (
                df_cleaned[col] > upper_bound
            )
            df_outliers = df_cleaned[outliers_mask]
            percent = round(len(df_outliers) / len(df_cleaned), 2) * 100

            df_cleaned = df_cleaned[~outliers_mask]

        print(
            f"Remaining DataFrame size: {len(df_cleaned)} and {percent} % of dataframe"
        )

        return df_cleaned

    def testing_adf(self, df: pd.DataFrame, adf_column):
        """
        Performs ADF test on selected column

        Parameters:
        - df: DataFrame containing the data to be tested
        - adf_column:  name of the column to perform the ADF test

        Returns:
        - adf_statistics
        - p_value
        """
        adf_test = adfuller(df[adf_column])
        adf_statistics = round(adf_test[0], 5)
        p_value = round(adf_test[1], 5)

        logger.info(f"adf_statistics: {adf_statistics }")
        logger.info(f"p_value: {p_value}")

        return adf_statistics, p_value

    def get_auto_arima(
        self, train_data: pd.DataFrame, temperature_series: str, seasonal: bool, m=None
    ):
        """
        Features ARIMA or SARIMA model to the training data.

        Parameters:
        - train_data: DataFrame with train set
        - temperature_series: name of the column with the temperature
        - seasonal: boolean indicating whether seasonal is True or False
        - m: the number of periods in each seasonal cycle; required if seasonal is True

        Returns:
        - model: ARIMA or SARIMA model
        """

        model = pm.auto_arima(
            train_data[temperature_series],
            seasonal=seasonal,
            m=m,
            d=0,
            start_p=0,
            start_q=0,
            max_p=7,
            max_q=7,
            trace=True,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
        )

        print(model.summary())

        return model

    def get_arima_forecast(
        self, model, initial_date: str, n_periods: int
    ) -> pd.DataFrame:
        """
        Generates a forecast and confidence intervals from the ARIMA or SARIMA model.

        Parameters:
        - model: ARIMA or SARIMA model
        - initial_date: the starting date (string) for the forecast
        - n_periods: the number of periods (months) to forecast into the future

        Returns:
        - forecast_df: DataFrame with forecast, confidence intervals and date

        """

        forecast, confidence_intervals = model.predict(
            n_periods=n_periods, return_conf_int=True
        )
        forecast_df = pd.DataFrame(columns=["Forecast", "Lower_CI", "Upper_CI"])
        forecast_df["Forecast"] = forecast
        forecast_df["Lower_CI"] = confidence_intervals[:, 0]
        forecast_df["Upper_CI"] = confidence_intervals[:, 1]
        forecast_df["dt"] = pd.date_range(
            start=initial_date, periods=n_periods, freq="MS"
        )

        return forecast_df

    def get_prophet_model(self, train_data: pd.DataFrame):
        """
        Features Prophet model to the training data

        Parameters:
        - train_data: DataFrame with train set

        Returns:
        - model: A trained Prophet model
        """

        train_data = train_data.copy()
        train_data["ds"] = train_data["dt"]
        train_data["y"] = train_data["AverageTemperatureByCountry"]
        model = Prophet(seasonality_mode="multiplicative", yearly_seasonality=True)
        model = model.fit(train_data)

        return model

    def get_prophet_forecast(
        self,
        model,
        test_data: pd.DataFrame,
        n_periods: str,
    ) -> pd.DataFrame:
        """
        Generates a forecast and confidence intervals from the Prophet model

        Parameters:
        - model: the Prophet model
        - train_data: DataFrame with train set
        - n_periods: the number of periods (months) to forecast into the future

        Returns:
        - test_forecast: DataFrame with forecast and confidence intervals for the test data
        - future_forecast: DataFrame with forecast and confidence intervals for the future periods
        """

        test_data = test_data.copy()
        test_data["ds"] = test_data["dt"]
        test_data["y"] = test_data["AverageTemperatureByCountry"]

        test_forecast = model.predict(test_data)
        future = model.make_future_dataframe(
            periods=n_periods, freq="MS", include_history=False
        )
        future_forecast = model.predict(future)

        test_forecast.rename(
            columns={
                "ds": "dt",
                "yhat": "Forecast",
                "yhat_lower": "Lower_CI",
                "yhat_upper": "Upper_CI",
            },
            inplace=True,
        )

        future_forecast.rename(
            columns={
                "ds": "dt",
                "yhat": "Forecast",
                "yhat_lower": "Lower_CI",
                "yhat_upper": "Upper_CI",
            },
            inplace=True,
        )

        return test_forecast, future_forecast

    def forecast_accuracy(self, forecast: np.ndarray, actual: np.ndarray):
        """
        Calculate accuracy metrics for forecast

        Parameters:
        - forecast: numpy array with forecasted values
        - actual: numpy array with actual values

        Returns:
        - Dictionary with accuracy metrics
        """

        mape = np.mean(np.abs(forecast - actual) / np.abs(actual))
        me = np.mean(forecast - actual)
        mae = np.mean(np.abs(forecast - actual))
        mpe = np.mean((forecast - actual) / actual)

        return {"mape": mape, "me": me, "mae": mae, "mpe": mpe}


class Visual:
    def __init__(
        self,
        linestyle="-",
        marker="o",
        color="b",
        title_fontsize=14,
        label_fontsize=12,
        grid=True,
    ):
        self.linestyle = linestyle
        self.marker = marker
        self.color = color
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.grid = grid

    def create_visuals_scatterplot(
        self,
        df: pd.DataFrame,
        time_series: str,
        temperature_series: str,
        fig=None,
        ax=None,
    ) -> tuple:
        """
        Creates a scatter plot of temperature over time.

        Parameters:
        - df: DataFrame with data to plot
        - time_series: column name with date
        - temperature_series: column name with values
        - fig: optional, matplotlib figure object to plot on. If none, a new figure is created.
        - ax: optional, matplotlib axes object to plot on. If none, new axes are created.

        Returns:
        - fig, ax: Tuple of matplotlib figure and axes objects with scatter plot
        """
        if time_series is None:
            raise ValueError('Obligatory parameter "time_series" is not given.')

        if temperature_series is None:
            raise ValueError('Obligatory parameter "temperature_series" is not given.')

        if (fig is None) or (ax is None):
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            df[time_series],
            df[temperature_series],
            marker=self.marker,
            linestyle=self.linestyle,
            color=self.color,
            label="Temperature",
        )
        ax.set_title(
            f'Temperature Over Time for {df["Country"].unique()}',
            fontsize=self.label_fontsize,
        )
        ax.set_xlabel("Time", fontsize=self.label_fontsize)
        ax.set_ylabel("Temperature", fontsize=self.label_fontsize)
        ax.grid(self.grid)

        return fig, ax

    def add_trending_line(
        self,
        df: pd.DataFrame,
        time_series: str,
        temperature_series: str,
        fig=None,
        ax=None,
    ) -> tuple:
        """
        Adds a linear trend line to a scatter plot of temperature over time.

        Parameters:
        - df: DataFrame with data to plot
        - time_series: column name with date
        - temperature_series: column name with values
        - fig: optional, matplotlib figure object to plot on. If none, a new figure is created.
        - ax: optional, matplotlib axes object to plot on. If none, new axes are created.

        Returns:
        - fig, ax: tuple of matplotlib figure and axes objects with scatter plot with trend line
        """
        if time_series is None:
            raise ValueError('Obligatory parameter "time_series" is not given.')

        if temperature_series is None:
            raise ValueError('Obligatory parameter "temperature_series" is not given.')

        if (fig is None) or (ax is None):
            fig, ax = plt.subplots(figsize=(10, 6))

        time_series_visual = (df[time_series] - df[time_series].min()).dt.days

        coefficient_p = np.polyfit(time_series_visual, df[temperature_series], 1)
        slope = coefficient_p[0]
        intercept = coefficient_p[1]
        ax.plot(
            df[time_series],
            slope * time_series_visual + intercept,
            linestyle=self.linestyle,
            color="r",
        )

        plt.legend()
        return fig, ax

    def create_forecast_visual(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        forecast: pd.DataFrame,
        time_series: str,
        temperature_series: str,
    ) -> tuple:
        """
        Creates plot for forecast, train and test data, including confidence intervals.

        Parameters:
        - train_data: DataFrame with training data
        - test_data: DataFrame with test data
        - forecast: DataFrame with forecast and  confidence intervals
        - time_series: column name with date
        - temperature_series: column name with temperature

        Returns:
        - fig, ax: tuple of matplotlib figure and axes objects with the forecast plot
        """

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            train_data[time_series],
            train_data[temperature_series],
            label="Train",
            color="red",
        )
        ax.plot(
            test_data[time_series],
            test_data[temperature_series],
            label="Test",
            color="green",
        )
        ax.plot(
            forecast[time_series],
            forecast["Forecast"],
            label="Forecast",
            color=self.color,
        )
        ax.fill_between(
            forecast[time_series],
            forecast["Lower_CI"],
            forecast["Upper_CI"],
            color="grey",
            alpha=0.15,
            label="Confidence Interval",
        )

        ax.set_title(
            "Temperature Forecast with Confidence Intervals",
            fontsize=self.label_fontsize,
        )
        ax.set_xlabel("Time", fontsize=self.label_fontsize)
        ax.set_ylabel("Temperature", fontsize=self.label_fontsize)
        ax.legend()
        ax.grid(self.grid)

        return fig, ax
