from datetime import datetime
import pandas as pd

def extract_date_dataframe(df):
    '''
    function to extract year, monthand hour from one column in data frame
    :param df: data frame
    :return new_dataframe: data frame with additional columns - year, month and hour
    '''
    year_dataframe = df.apply(lambda d: pd.Series({'year': datetime.strptime(d, "%Y-%m-%d %H:%M:%S").year}))
    month_dataframe = df.apply(lambda d: pd.Series({'month': datetime.strptime(d, "%Y-%m-%d %H:%M:%S").month}))
    #date_dataframe = df.apply(lambda d: pd.Series({'date': datetime.strptime(d, "%Y-%m-%d %H:%M:%S").day}))
    hour_dataframe = df.apply(lambda d: pd.Series({'hour': datetime.strptime(d, "%Y-%m-%d %H:%M:%S").hour}))

    new_dataframe = pd.concat([df,year_dataframe,month_dataframe,hour_dataframe], axis=1)

    return new_dataframe



