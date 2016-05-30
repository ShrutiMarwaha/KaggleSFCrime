# import libraries
import pandas as pd
from sklearn import cross_validation as cv
from processors import modeling
from processors import aggregate_classes as ac

# load training set features
training_features = pd.read_pickle("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/training_features.pkl")
outcomes = pd.read_pickle("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/outcomes.pkl")

# aggregate crime categories into bigger classes
crime_class = []

for outcome in outcomes:
    crime_class.append( ac.get_key(ac.crime_dict,outcome) )

crime_class = pd.Series(crime_class)
print crime_class.shape
print crime_class.value_counts(ascending=True)

# Create Dummy Variables from Categorical Data
# decide which columns should be categorical and converted to dummy variables. this step cannot be automated, pay attention !!
categorical_columns = training_features.columns.tolist()
training_dummy_var = modeling.create_dummy_var(training_features,categorical_columns)

# # divide data in to training and intermediate set
features_train, features_intermediate, outcomes_train, outcomes_intermediate = cv.train_test_split(training_dummy_var,crime_class,test_size=0.4,random_state=0)
# # divide intermediate set into test and validation set.
# # validation set will be only used once to evaluate final model's performance
features_test, features_validation, outcomes_test, outcomes_validation = cv.train_test_split(features_intermediate,outcomes_intermediate,test_size=0.5,random_state=0)

print "build model through function \n"
modeling.basic_model("LogisticRegression",features_train,outcomes_train,features_test,outcomes_test)
# modeling.basic_model("RandomForestClassifier",features_train,outcomes_train,features_test,outcomes_test)
# modeling.basic_model("BernoulliNB",features_train,outcomes_train,features_test,outcomes_test)
# modeling.basic_model("GradientBoostingClassifier",features_train,outcomes_train,features_test,outcomes_test) # takes very long
# modeling.basic_model("SVC",features_train,outcomes_train,features_test,outcomes_test) # donot try, takes very very long
