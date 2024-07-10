import pandas as pd
import numpy as np
import logging as logger
from collections import Counter


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
            logger.warning(f'{e} error. File at {file_path} is not csv')
            return pd.DataFrame()

    def parse_dates(self, df: pd.DataFrame, date_list:list = [])-> pd.DataFrame:
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

    def merge_data(self, df_1: pd.DataFrame, df_2:pd.DataFrame, merge_columns:list, suffix:str)-> pd.DataFrame:
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
                df_2 = df_2.rename(columns={column: f'{column}{suffix}'})

        if df_1.empty:
            return df_2

        if df_2.empty:
            return df_1

        for item in merge_columns:
            if item not in df_2.columns:
                df_name = f"{df_2=}".split('=')[0]
                logger.warning(f'{df_name} does not contain {item} and cannot be merged')
                return df_1

        df_total = pd.merge(df_1, df_2, on=merge_columns, how='outer')
        return df_total

class CheckData:
    def __init__(self, continents, countries):
        self.continents = continents
        self.countries = countries

    def check_nans(self, df:pd.DataFrame) -> bool:
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
            print(f'Check NaNs - False. Columns with NaNs: {columns_with_nans}')
            return False
        else:
            print('Check NaNs - True: dataset does not contain any NaN values')
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
        df['Continent_in_Country'] = df['Country'].isin(self.continents)
        count = df['Continent_in_Country'].sum()
        error_percent = round((count / len(df) ) * 100, 2)
        logger.warning(f'{count} entires have continent in country. It is {error_percent} % of the entire dataset ')
        return df


    def correct_data_in_columns(self, df: pd.DataFrame, corrections: dict, column: str) -> pd.DataFrame:

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

        logger.warning(f'Columns changed: {dict(col_changes_count)}') #TODO
        df[column] = df[column].replace(corrections)
        return df

    def correct_country_names(self, df: pd.DataFrame)-> pd.DataFrame:
        """
            Corrects country names in a DataFrame using the `countries` dictionary

            Parameters:
            - df: DataFrame with data to be corrected

            Returns:
            - df: DataFrame with corrected data in 'Country' column
        """
        df = self.correct_data_in_columns(df, self.countries, 'Country')
        return df

class Model:
    def __init__(self, modeling_columns):
        self.modeling_columns = modeling_columns

    def get_modeling_columns(self, df:pd.DataFrame, modeling_columns: list) -> pd.DataFrame:
        modeling_df = df.loc[:,modeling_columns]
        nans = modeling_df[modeling_columns].isna().sum().sum()
        nans_percent = round(nans / len(modeling_df) * 100, 2)
        if nans_percent < 10:
            logger.info(f'LOW number of not existent values in modeling column {modeling_columns}: {nans_percent}%')
        else:
            logger.warning((f'HIGH number of non existent values in modeling column {modeling_columns}: {nans_percent}%'))
        return modeling_df

    def drop_nans(self, df:pd.DataFrame, modeling_columns: list) ->pd.DataFrame:
        df.dropna(subset= modeling_columns, inplace=True)
        return df

    def test_train_split(self, df:pd.DataFrame) -> pd.DataFrame:
        mask = np.random.rand(len(df)) < 0.8
        train = df[mask]
        test = df[~mask]
        return train, test

    #czy będzie padać (klasyfikacja)
    # przewidywanie temperatury Arima (regresja)
    #metoda:
        #biorę tylko kolumny, które są potrzebne do modelowania
        #robię kopię tych danych
        #spradzić, ale procentowo jest NaNs w wybranych kolumnach
    #klasa vizualizacja
    #pyplot
     #dane przed modelowaniem
     #na osi X czas, na osi Y temperatura

    #test/ train - splitujemy
    #test - 30%
    #train - 70%























