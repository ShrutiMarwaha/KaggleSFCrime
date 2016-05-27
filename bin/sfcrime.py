# import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn import grid_search as gs
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
outcomes.shape
outcomes_frequency = outcomes.value_counts(ascending=True)
outcomes_frequency.head()

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

###################### Grid Search with Cross Validation ######################
# to run k-fold cross validation, remove classes which have less than "k" samples.
# cv< minimum no.of samples in each class. So either remove such samples or use KFold
outcomes_frequency.head()
outcomes_frequency[outcomes_frequency < 10 ]
matchings_indices = [ i for i, value in enumerate(outcomes) if value == "TREA" ]
# for i, value in enumerate(outcomes):
#             if value == "TREA":
#                 print i
outcomes[matchings_indices]
# remove these samples from features and outcomes data
outcomes2 = outcomes.drop(matchings_indices)
training_dummy_var2 = training_dummy_var.drop(matchings_indices)

# divide data in to training and validation set. Since performing cross validation, no need for test set
# validation set will be only used once to evaluate final model's performance
features_train, features_validation, outcomes_train, outcomes_validation = cv.train_test_split(training_dummy_var2,outcomes2,test_size=0.4,random_state=0)

# divide data in to intermediate set into test and validation set. validation set will be only used once to evalaute final model's performance
# features_test, features_validation, outcomes_test, outcomes_validation = cv.train_test_split(features_intermediate,outcomes_intermediate,test_size=0.5,random_state=0)
print(features_train.shape)
print(features_validation.shape)

# select the machine learning algorithm you want to apply
algo = LogisticRegression(random_state=0,n_jobs=-1)
# algo = RandomForestClassifier(random_state=0,n_jobs=-1)
# algo = BernoulliNB()
# algo = GradientBoostingClassifier()

# param_grid is dictionary where key is parameter name and value is the numeric values you want to try for that parameter
# parameter grid for Logistic Regression
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
# # parameter grid for Random Forest
# param_grid = {'n_estimators': [10, 100, 200], 'max_depth': [None,15,30], 'max_features': ['sqrt','log2']}
# # parameter grid for Naive Bayes
# param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
# # parameter grid for Gradient Boosting Classifier
# param_grid = {'learning_rate': [0.1,1,10], 'max_depth': [3,10,15], 'n_estimators': [100, 500, 1000], 'max_features': ['sqrt','log2']}

model = gs.GridSearchCV(algo, param_grid, cv=3, scoring='accuracy')
# TODO: model = gs.GridSearchCV(algo, param_grid, cv=3, scoring='f1_samples',n_jobs=-1) ; gives error
# scoring: f1_samples (for multilabel sample); f1 (for for binary targets); accuracy (for model accuracy)
model.fit(features_train, outcomes_train)

# model scores for each parameter used in grid
print model.grid_scores_
# examine the best model
print model.best_score_
print model.best_params_
# print model.best_estimator_

# now create model with the best parameters.
# this step can be avoided if all arguments as used in GridSearchCV. GridSearchCV automatically refits the best model.
# But GridSearch with 10 fold CV takes very long.
model = LogisticRegression(solver='lbfgs',multi_class='multinomial',n_jobs=-1,random_state=0,C=1)
# model = RandomForestClassifier(n_estimators=200,max_depth=15,n_jobs=-1,random_state=0)
# model = BernoulliNB(alpha=300)
# TODO: model = GradientBoostingClassifier(random_state=0)

# make predictions.
predicted = model.predict(features_test)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

# use RandomizedSearchCV searches a subset of the parameters to reduce computational expense
