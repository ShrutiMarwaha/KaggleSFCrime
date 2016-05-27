import pandas as pd

def load_csv_data(filename,n_rows=None):
    """
    reads csv file and returns data-frame using pandas
    :param filename: csv file
    :return: data-frame (pandas)
    """

    output = pd.read_csv(filename,nrows=n_rows)
    return(output)