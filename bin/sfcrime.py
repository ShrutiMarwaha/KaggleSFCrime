# import local functions/processors
from processors import loader
import analysis
from processors import feature_extractor as extractor
from processors import feature_engineering as engineering
from processors import modeling

# import libraries
from sklearn import cross_validation as cv
from sklearn import metrics
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB

# load data
training_set = loader.load_csv_data("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/Datasets/train.csv")
test_set = loader.load_csv_data("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/Datasets/test.csv")

# data exploration
analysis.data_summary(training_set)
analysis.data_summary(test_set)

# outcomes: classes to be predicted
outcomes = training_set.Category
# convert outcome classes from string to numeric
# le_class = preprocessing.LabelEncoder()
# outcomes = le_class.fit_transform(training_set.Category)
# outcomes = pd.Series(outcomes)
# save files for future use
# outcomes.to_pickle("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/outcomes.pkl")
outcomes_frequency = outcomes.value_counts(ascending=True)
print "number of samples in each class: \n%s" % outcomes_frequency.head()

# feature extraction
training_set = extractor.extract_date_time(training_set)
test_set = extractor.extract_date_time(test_set)

# feature engineering
training_set = engineering.find_zipcodes_dataframe(training_set)
test_set = engineering.find_zipcodes_dataframe(test_set)

print "training set:\n%s" % training_set.head(3)
training_features = training_set.drop(["Category","Dates","Resolution","Descript","Address","X","Y"], axis=1)
# remove the columns that should not be used for model building
print "training set:\n%s" % training_features.head(3)
test_features = test_set.drop(["Dates","Address","X","Y"], axis=1)

# save files for future use
# training_features.to_pickle("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/training_features.pkl")
# test_features.to_pickle("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/test_features.pkl")

# load training set features
# training_features = pd.read_pickle("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/training_features.pkl")
# outcomes = outcomes = pd.read_pickle("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/outcomes.pkl")

# decide which columns should be categorical and converted to dummy variables. this step cannot be automated, pay attention !!
train_categorical_columns = list(training_features)
print train_categorical_columns
# Create Dummy Variables from Categorical Data
training_dummy_var = modeling.create_dummy_var(training_features,train_categorical_columns)
analysis.data_summary(training_dummy_var)

test_categorical_columns = list(test_features)
print test_categorical_columns
#do not include first column - ID
test_categorical_columns.remove("Id")
test_dummy_var = modeling.create_dummy_var(test_features.drop(["Id"], axis=1),test_categorical_columns)
analysis.data_summary(test_dummy_var)

# divide data in to training and intermediate set
features_train, features_intermediate, outcomes_train, outcomes_intermediate = cv.train_test_split(training_dummy_var,outcomes,test_size=0.4,random_state=0)
# divide intermediate set into test and validation set.
# validation set will be only used once to evaluate final model's performance
features_test, features_validation, outcomes_test, outcomes_validation = cv.train_test_split(features_intermediate,outcomes_intermediate,test_size=0.5,random_state=0)
print "dimension of training set %s" % (features_train.shape,)
print "dimension of test set %s" % (features_test.shape,)
print "dimension of validation set %s" % (features_validation.shape,)

# build the model
modeling.basic_model("LogisticRegression",features_train,outcomes_train,features_test,outcomes_test)
modeling.basic_model("RandomForestClassifier",features_train,outcomes_train,features_test,outcomes_test)
modeling.basic_model("BernoulliNB",features_train,outcomes_train,features_test,outcomes_test)
# modeling.basic_model("GradientBoostingClassifier",features_train,outcomes_train,features_test,outcomes_test) # takes very long
# modeling.basic_model("SVC",features_train,outcomes_train,features_test,outcomes_test) # donot try, takes very very long


###################### Grid Search with Cross Validation ######################
# to run k-fold cross validation, remove classes which have less than "k" samples.
# cv< minimum no.of samples in each class. So either remove such samples or use KFold
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
print(cv_features_train.shape)
print(cv_features_validation.shape)

# GridSearch with 10 fold CV will take very long, so running only 3 folds to find the best parameters.
# use RandomizedSearchCV that searches a subset of the parameters to reduce computational expense
# Look at modeling.py for parameters to vary in grid search
modeling.gridsearch_cv_model("LogisticRegression",2,cv_features_train,cv_outcomes_train)
modeling.gridsearch_cv_model("RandomForestClassifier",2,cv_features_train,cv_outcomes_train)
modeling.gridsearch_cv_model("BernoulliNB",2,cv_features_train,cv_outcomes_train)
# modeling.gridsearch_cv_model("GradientBoostingClassifier",2,cv_features_train,cv_outcomes_train)
# modeling.gridsearch_cv_model("SVC",2,cv_features_train,cv_outcomes_train)

# now chose algorithm with the best parameters.
# this step can be avoided if all desired arguments are used in GridSearchCV. GridSearchCV automatically refits the best model.
model = LogisticRegression(solver='lbfgs',multi_class='multinomial',C=1,n_jobs=-1,random_state=0)
# model = RandomForestClassifier(n_jobs=-1,random_state=0)
# model = BernoulliNB(alpha=300)
# TODO: model = GradientBoostingClassifier(random_state=0)
# TODO: model = SVC(random_state=0)
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
