import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn import metrics

from sklearn import preprocessing
# local functions/processors
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

le_class = preprocessing.LabelEncoder()
crime = le_class.fit_transform(training_features.Category)



