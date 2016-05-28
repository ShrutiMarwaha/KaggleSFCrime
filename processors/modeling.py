import pandas as pd

def create_dummy_var(df,categorical_col):
    """
    Create Dummy Variables from Categorical Data
    :param df, categorical_col: data frame, columns which should be converted to dummy variables
    :return dummy_var_df: dummy variable data frame
    """
    if len(categorical_col) > 0:
        for i in df.columns:
            df[i] = df[i].astype('category')

    dummy_var_df = pd.get_dummies(df)
    return dummy_var_df

def basic_model(model_name):
    pass

def model1():
    pass