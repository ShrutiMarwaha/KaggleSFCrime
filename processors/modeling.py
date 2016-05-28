import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB


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

def basic_model(mlalgo,train_variables,train_output,test_variables,test_output):
    if mlalgo=="BernoulliNB":
        model = BernoulliNB()
    elif mlalgo=="SVC":
        model = SVC()
    elif mlalgo=="GradientBoostingClassifier":
        model = GradientBoostingClassifier(random_state=0)
    elif mlalgo=="RandomForestClassifier":
        model=RandomForestClassifier(n_jobs=-1,random_state=0)
    else:
        model = LogisticRegression(n_jobs=-1,random_state=0)

    # build the model
    model.fit(train_variables, train_output)

    expected = test_output
    # make predictions
    predicted = model.predict(test_variables)

    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    # print(metrics.roc_auc_score(expected, predicted)) # predicted outputs have to be binarized
    return metrics.accuracy_score(expected, predicted)

# def basic_model(mlalgo,train_variables,train_output,test_variables,test_output):
#     if mlalgo=="BernoulliNB":
#         model = BernoulliNB()
#     elif mlalgo=="SVC":
#         model = SVC()
