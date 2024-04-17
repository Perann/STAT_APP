""" 
In this file, we will preprocess the data for the web app.
The aim of making it here in another file is to avoid running at each we change page or simply relaod it.

The preprocessing will be done in the manage.py file, which is the first file to be run when we start the server.
"""

# Importing packages
import pandas as pd
import os

def tweak_alternative_data(df_):
    return (df_.assign(**{col + '_%y/y': df_[col].pct_change(fill_method = None) for col in df_.columns}) #Adding the returs
            .set_index(pd.to_datetime(df_.index)) #Changing the index format to datetime
            .rename(lambda c: c.replace(' ','_'), axis = 1)) # Replacing the spaces by _ in the names


def preprocessing():
    alternative_data_raw = pd.read_excel('./original_input/EnsaeAlternativeTimeSeries.xlsx', index_col=0)
    alternative_data = tweak_alternative_data(alternative_data_raw)
    alternative_data = alternative_data[[col for col in alternative_data.columns if '%y/y' in col]]
    alternative_data.to_excel('./apps/tmp_data_return.xlsx')


if __name__ == '__main__':
    preprocessing()
    os.remove('./apps/tmp_data_return.xlsx')