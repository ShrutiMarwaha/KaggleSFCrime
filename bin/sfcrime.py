# import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation as cv
from sklearn import metrics
from sklearn import preprocessing

# import local functions/processors
from processors import loader
from processors import feature_extractor as extractor
from processors import feature_engineering as engineering
from processors import modeling
from sklearn import preprocessing

# load data
training_set = loader.load_csv_data("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/train.csv")
test_set = loader.load_csv_data("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/test.csv")

# data exploration
print "training set: \n %s \n" % training_set.head()
print "dimension of training set: %s \n" % (training_set.shape,)
print "dimension of training set without missing data %s \n" %  (training_set.dropna().shape,)
print "column names of training set: %s \n" % training_set.columns
print "first 5 rows of column Category of training set: \n %s \n" % training_set.Category.head()
#print training_set.Category.value_counts()

print "dimension of test data: %s \n" % (test_set.shape,)
print "test data: \n %s \n" % test_set.head()

# feature extraction
training_striped_time = extractor.extract_date_dataframe(training_set.Dates)
test_striped_time = extractor.extract_date_dataframe(test_set.Dates)

# feature engineering
training_zipcodes = training_set.apply(lambda d: engineering.closest_zipcode(d["X"],d["Y"]), axis=1)
training_zipcodes.name = "zip"
print len(training_zipcodes.unique())

test_zipcodes = test_set.apply(lambda d: engineering.closest_zipcode(d["X"],d["Y"]), axis=1)
test_zipcodes.name = "zip"
#training_zipcodes.to_pickle("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/training_zipcodes.pkl")

# combine important features
training_features = pd.concat( (training_striped_time,training_zipcodes,training_set[["DayOfWeek","PdDistrict","Category"]]), axis=1)
print "after features extraction - training data: \n %s \n" % training_features.head()
test_features = pd.concat( (test_striped_time,test_zipcodes,test_set[["DayOfWeek","PdDistrict"]]), axis=1)
print "after features extraction - training datatest data: \n %s \n" % test_features.head()

# save files for future use
# training_features.to_pickle("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/training_features.pkl")
# test_features.to_pickle("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/test_features.pkl")

# load training set features
# training_features = pd.read_pickle("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/training_features.pkl")

# Create Dummy Variables from Categorical Data
# remove outcomes, keep features only
training_features2 = training_features.drop("Category", axis=1)
training_features2.dtypes
# decide which columns should be categorical and converted to dummy variables. this step cannot be automated, pay attention !!
categorical_columns = training_features2.columns.tolist()

training_dummy_var = modeling.create_dummy_var(training_features2,categorical_columns)
print "compare number of features earlier: %s, now: %s" % (training_features2.shape[1], training_dummy_var.shape[1])
print training_dummy_var.columns.tolist()
print training_dummy_var.head()

outcomes = training_features.Category
# convert outcome classes from string to numeric
# le_class = preprocessing.LabelEncoder()
# outcomes = le_class.fit_transform(training_features.Category)

# divide data in to training and intermediate set
features_train, features_intermediate, outcomes_train, outcomes_intermediate = cv.train_test_split(training_dummy_var,outcomes,test_size=0.4,random_state=0)
# divide intermediate set into test and validation set.
# validation set will be only used once to evaluate final model's performance
features_test, features_validation, outcomes_test, outcomes_validation = cv.train_test_split(features_intermediate,outcomes_intermediate,test_size=0.5,random_state=0)
print(features_train.shape)
print(features_test.shape)
print(features_validation.shape)

# build the model
model = LogisticRegression(n_jobs=-1,random_state=0)
#model = RandomForestClassifier(n_jobs=-1,random_state=0)
#model = BernoulliNB()
#model = SVC() # donot try, takes very very long
#model= GradientBoostingClassifier(random_state=0) # takes very long
model.fit(features_train, outcomes_train)

# make predictions
expected = outcomes_test
predicted = model.predict(features_test)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
print(metrics.roc_auc_score(expected, predicted)) # predicted outputs have to be binarized
print(metrics.accuracy_score(expected, predicted))


