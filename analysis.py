import numpy as np

def data_summary(df):
    '''
    print basic head, dimension, column names
    :param df: data frame
    :return: dimension, missing data, column names, top 3 lines
    '''
    print "dataset dimensions: %s \n" % (df.shape,)
    print "number of rows without missing data: %s \n" % np.count_nonzero(df.isnull())
    print "columns of dataset: %s \n" % df.columns.tolist()
    print "first 3 rows: \n%s" % df.iloc[:3,:]



