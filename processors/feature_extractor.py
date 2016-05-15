from datetime import datetime
import pandas as pd

# function to extract year, month, date and hour from one column in dataframe
def extract_date_dataframe(data_frame):
    year_dataframe = data_frame.apply(lambda d: datetime.strptime(d, "%Y-%m-%d %H:%M:%S").year)
    month_dataframe = data_frame.apply(lambda d: datetime.strptime(d, "%Y-%m-%d %H:%M:%S").month)
    date_dataframe = data_frame.apply(lambda d: datetime.strptime(d, "%Y-%m-%d %H:%M:%S").day)
    hour_dataframe = data_frame.apply(lambda d: datetime.strptime(d, "%Y-%m-%d %H:%M:%S").hour)

    new_dataframe = pd.concat([year_dataframe,month_dataframe,date_dataframe,hour_dataframe], axis=1)
    new_dataframe.columns = ["year","month","date","hour"]

    return new_dataframe

