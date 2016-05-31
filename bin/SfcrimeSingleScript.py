# this is same script as sfcrime.py, all the required functions are defined in a single file.

# import libraries
import pandas as pd
import numpy as np
from datetime import datetime
from pyzipcode import ZipCodeDatabase
from scipy.spatial import distance
from sklearn import cross_validation as cv
from sklearn import metrics
from sklearn import grid_search as gs
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB

############################################ define all functions ############################################
def data_summary(df):
    '''
    print basic head, dimension, columnames
    :param df: data frame
    :return: dimension, missing data, column names, top 3 lines
    '''
    print "dataset dimensions: %s \n" % (df.shape,)
    print "number of rows without missing data: %s \n" % np.count_nonzero(df.isnull())
    print "columns of dataset: %s \n" % df.columns.tolist()
    print "first 3 rows: \n%s" % df.iloc[:3,:]

def extract_date_time(data_frame,colname="Dates"):
    '''
    function to extract year, month and hour from one column in data frame
    :param data_frame, colname: data frame, column name which contains datetime
    :return new_dataframe: original data frame with additional columns - year, month and hour
    '''
    data_col = data_frame[colname]

    year_dataframe = data_col.apply(lambda d: pd.Series({'year': datetime.strptime(d, "%Y-%m-%d %H:%M:%S").year}))
    month_dataframe = data_col.apply(lambda d: pd.Series({'month': datetime.strptime(d, "%Y-%m-%d %H:%M:%S").month}))
    #date_dataframe = data_col.apply(lambda d: pd.Series({'date': datetime.strptime(d, "%Y-%m-%d %H:%M:%S").day}))
    hour_dataframe = data_col.apply(lambda d: pd.Series({'hour': datetime.strptime(d, "%Y-%m-%d %H:%M:%S").hour}))

    new_dataframe = pd.concat([data_frame,year_dataframe,month_dataframe,hour_dataframe], axis=1)

    return new_dataframe

zcdb = ZipCodeDatabase()
sf_zipcodes = zcdb.find_zip(city="San Francisco", state="CA")
# print(len(sf_zipcodes))

def long_lat_to_zipcode(input_longitude,input_latitude):
    """
    function to convert latitude and longitude to zipcode:
    find euclidean distance between user provided latitude and longitude and
    all latitudes, longitudes from sf_zipcodes and select the row from latter
    which has minimum distance. then extract its zip code

    :param input_longitude,input_latitude: latitude and longitude that you want to convert to closest zipcode
    :return closest_zip: zip code
    """

    closest_zip = None
    euc_dist = None

    for i, sf_zipcode in enumerate(sf_zipcodes):
        # extract latitude and longitude from each row in sf_zipcode
        lat = sf_zipcode.latitude
        long = sf_zipcode.longitude
        # calculate euclidean distance between lat and long from sf_zipcode and input latitude and longitude value
        euclidean_dist = round( distance.euclidean((long,lat), (input_longitude,input_latitude)), 4)
        # assign the euclidean distance calculated for first row as euc_dist
        if i == 0:
            euc_dist = euclidean_dist

        # if the euclidean_dist is smaller than the one calculated in previous iteration,
        # replace it with current one and extract the corresponding zipcode
        if euclidean_dist < euc_dist:
            euc_dist = euclidean_dist
            closest_zip = int(sf_zipcode.zip)

    return closest_zip

def find_zipcodes_dataframe(data_frame,long_col="X",lat_col="Y"):
    '''
    function that applies long_lat_to_zipcode function to all rows of data frame
    :param data_frame: data frame
    :param long_col: column name that contains longitude
    :param lat_col: column name that contains latitude
    :return data_frame: original data frame with additional column containing zipcodes
    '''

    # output = data_frame.apply(lambda d: long_lat_to_zipcode(d[long_col],d[lat_col]), axis=1)
    data_frame.loc[:,"zip"] = pd.Series( data_frame.apply(lambda d: long_lat_to_zipcode(d[long_col],d[lat_col]), axis=1), index=data_frame.index)
    return data_frame

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

def basic_model(mlalgo,trainingset_features,trainingset_outcomes,testset_features,testset_outcomes):
    '''
    simple function to apply machine learning algorithms with default parameters.
    random_state is used to set pseudo seed, n_jobs=-1 uses all available cpus

    :param mlalgo: machine learning algorithms
    :param trainingset_features:
    :param trainingset_outcomes:
    :param testset_features:
    :param testset_outcomes:
    :return:
    '''
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
    model.fit(trainingset_features, trainingset_outcomes)

    expected = testset_outcomes
    # make predictions
    predicted = model.predict(testset_features)
    predicted_prob = model.predict_proba(testset_features)
    loss = metrics.log_loss(expected, predicted_prob)

    # summarize the fit of the model
    print("accuracy score: %s \n" % metrics.accuracy_score(expected, predicted))
    print("classification_report: \n%s" % metrics.classification_report(expected, predicted))
    print("confusion matrix: \n%s" % metrics.confusion_matrix(expected, predicted))
    print("log loss: %s \n" % loss)
    #print("auc score: %s \n" % metrics.roc_auc_score(expected, predicted)) # predicted outputs have to be binarized
    return metrics.accuracy_score(expected, predicted)

def gridsearch_cv_model(mlalgo,folds,trainingset_features,trainingset_outcomes):
    '''
    Grid Search with Cross Validation.
    param_grid is dictionary where key is parameter name and value is the numeric values you want to try for that parameter

    :param mlalgo: machine learning algorithm to apply
    :param folds: cross validations folds
    :param trainingset_features:
    :param trainingset_outcomes:
    :return cv_model.best_params_: parameters from param_grid which give best model scores
    '''
    if mlalgo=="BernoulliNB":
        algo = BernoulliNB()
        param_grid = {'alpha': [0.1, 1, 10]}
    elif mlalgo=="SVC":
        algo = SVC()
        param_grid = {'C': [0.1, 1, 10]}
    elif mlalgo=="GradientBoostingClassifier":
        algo = GradientBoostingClassifier(random_state=0)
        #param_grid = {'learning_rate': [0.1,1,10], 'max_depth': [3,10,15], 'n_estimators': [100, 500, 1000], 'max_features': ['sqrt','log2']}
        param_grid = {'learning_rate': [0.1,1,10]}
    elif mlalgo=="RandomForestClassifier":
        algo=RandomForestClassifier(n_jobs=-1,random_state=0)
        param_grid = {'n_estimators': [10, 100, 200], 'max_depth': [None,15,30], 'max_features': ['sqrt','log2']}
    else:
        algo = LogisticRegression(n_jobs=-1,random_state=0)
        #param_grid = {'C': [0.001, 0.1, 1, 10, 100]}
        param_grid = {'C': [0.01, 1]}

    cv_model = gs.GridSearchCV(algo, param_grid, cv=folds, scoring='accuracy')
    cv_model.fit(trainingset_features, trainingset_outcomes)
    print "scores for each model %s" % cv_model.grid_scores_
    # examine the best model
    print "best parameter value %s" % cv_model.best_params_
    print "score for best model %s" % cv_model.best_score_

    return cv_model.best_params_


########################################################################################
# load data
training_set = pd.read_csv("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/Datasets/train.csv")
test_set = pd.read_csv("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/Datasets/test.csv")

# data exploration
data_summary(training_set)
data_summary(test_set)

# outcomes: classes to be predicted
outcomes = training_set.Category

# feature extraction
training_set = extract_date_time(training_set)
test_set = extract_date_time(test_set)

# feature engineering
training_set = find_zipcodes_dataframe(training_set)
test_set = find_zipcodes_dataframe(test_set)

# remove the columns that should not be used for model building
training_features = training_set.drop(["Category","Dates","Resolution","Descript","Address","X","Y"], axis=1)
test_features = test_set.drop(["Dates","Address","X","Y"], axis=1)

# decide which columns should be categorical and converted to dummy variables. this step cannot be automated, pay attention !!
train_categorical_columns = list(training_features)
print train_categorical_columns
# Create Dummy Variables from Categorical Data
training_dummy_var = create_dummy_var(training_features,train_categorical_columns)
data_summary(training_dummy_var)

test_categorical_columns = list(test_features)
print test_categorical_columns
#do not include first column - ID
test_categorical_columns.remove("Id")
test_dummy_var = create_dummy_var(test_features.drop(["Id"], axis=1),test_categorical_columns)
data_summary(test_dummy_var)

# divide data in to training and intermediate set
features_train, features_intermediate, outcomes_train, outcomes_intermediate = cv.train_test_split(training_dummy_var,outcomes,test_size=0.4,random_state=0)
# divide intermediate set into test and validation set.
# validation set will be only used once to evaluate final model's performance
features_test, features_validation, outcomes_test, outcomes_validation = cv.train_test_split(features_intermediate,outcomes_intermediate,test_size=0.5,random_state=0)

# build the model
basic_model("LogisticRegression",features_train,outcomes_train,features_test,outcomes_test)
basic_model("RandomForestClassifier",features_train,outcomes_train,features_test,outcomes_test)
basic_model("BernoulliNB",features_train,outcomes_train,features_test,outcomes_test)
# basic_model("GradientBoostingClassifier",features_train,outcomes_train,features_test,outcomes_test) # takes very long

###################### Grid Search with Cross Validation ######################
# to run k-fold cross validation, remove classes which have less than "k" samples.
# cv< minimum no.of samples in each class. So either remove such samples or use KFold
outcomes_frequency = outcomes.value_counts(ascending=True)
outcomes_frequency.head()
outcomes_frequency[outcomes_frequency < 10 ]
matchings_indices = [ i for i, value in enumerate(outcomes) if value == "TREA" ]
# matchings_indices = [ i for i, value in enumerate(outcomes) if value == 33 ] #if outcomes very converted to numeric
outcomes[matchings_indices]
# remove these samples from features and outcomes data
outcomes2 = outcomes.drop(matchings_indices)
training_dummy_var2 = training_dummy_var.drop(matchings_indices)

# divide data in to training and validation set. Since performing cross validation, no need for test set
# validation set will be only used once to evaluate final model's performance
cv_features_train, cv_features_validation, cv_outcomes_train, cv_outcomes_validation = cv.train_test_split(training_dummy_var2,outcomes2,test_size=0.4,random_state=0)

# GridSearch with 10 fold CV will take very long, so running only 3 folds to find the best parameters.
# use RandomizedSearchCV that searches a subset of the parameters to reduce computational expense
# Look at modeling.py for parameters to vary in grid search
gridsearch_cv_model("LogisticRegression",2,cv_features_train,cv_outcomes_train)
gridsearch_cv_model("RandomForestClassifier",2,cv_features_train,cv_outcomes_train)
gridsearch_cv_model("BernoulliNB",2,cv_features_train,cv_outcomes_train)
# gridsearch_cv_model("GradientBoostingClassifier",2,cv_features_train,cv_outcomes_train)

# now chose algorithm with the best parameters.
# this step can be avoided if all desired arguments are used in GridSearchCV. GridSearchCV automatically refits the best model.
model = LogisticRegression(solver='lbfgs',multi_class='multinomial',C=1,n_jobs=-1,random_state=0)
# model = RandomForestClassifier(n_jobs=-1,random_state=0)
# model = BernoulliNB(alpha=300)
# TODO: model = GradientBoostingClassifier(random_state=0)
model.fit(cv_features_train, cv_outcomes_train)

# make predictions on validation set. use only once to evaluate final model's performance
expected = cv_outcomes_validation
predicted = model.predict(cv_features_validation)
predicted_prob = model.predict_proba(cv_features_validation)
loss = metrics.log_loss(expected, predicted_prob)
model.classes_
result = pd.DataFrame(predicted_prob, columns=model.classes_)

# summarize the fit of the model
print("accuracy score: %s \n" % metrics.accuracy_score(expected, predicted))
print("classification_report: %s \n" % metrics.classification_report(expected, predicted))
print("confusion matrix: %s \n" % metrics.confusion_matrix(expected, predicted))
print("log loss: %s \n" % loss)

######### final code for submission to kaggle #######
#model = LogisticRegression(solver='lbfgs',multi_class='multinomial',C=1,n_jobs=-1,random_state=0)
model = RandomForestClassifier(n_jobs=-1,random_state=0)
model.fit(training_dummy_var, outcomes)

# make predictions
predicted_prob = model.predict_proba(test_dummy_var)
result = pd.DataFrame(predicted_prob, columns=model.classes_)
finalresult = pd.concat([ test_set.Id,result ], axis=1)

pd.set_option('display.max_columns', None)
print finalresult.head()
print finalresult.shape

finalresult.to_csv("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/Datasets/finalresult.csv")
