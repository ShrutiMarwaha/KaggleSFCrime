# Since classifying a crime into 39 categories is very challenging, Merge different categories of crimes into bigger classes.
# 3 major classes of crimes by law: Felony, Misdemeanor, Infraction
# 2 other classes of crimes: Violent crimes, Non-violent crimes

# import libraries
import pandas as pd
from sklearn import cross_validation as cv
from processors import modeling
from processors import aggregate_classes as ac
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics

# load training set features
training_features = pd.read_pickle("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/training_features.pkl")
outcomes = pd.read_pickle("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/outcomes.pkl")

# aggregate crime categories into bigger classes
crime_class = []

for outcome in outcomes:
    crime_class.append( ac.get_key(ac.crime_dict,outcome) ) # group the existing categories of crime into 3: Infraction, Misdemeanor, Felony
    #crime_class.append( ac.get_key(ac.crime_voilence_dict,outcome) ) # group the existing categories of crime into 2: Voilent and non-voilent crimes


crime_class = pd.Series(crime_class)
print crime_class.head()

# Create Dummy Variables from Categorical Data
# decide which columns should be categorical and converted to dummy variables. this step cannot be automated, pay attention !!
categorical_columns = training_features.columns.tolist()
training_dummy_var = modeling.create_dummy_var(training_features,categorical_columns)

# divide data in to training and intermediate set
features_train, features_intermediate, outcomes_train, outcomes_intermediate = cv.train_test_split(training_dummy_var,crime_class,test_size=0.4,random_state=0)
# divide intermediate set into test and validation set.
# validation set will be only used once to evaluate final model's performance
features_test, features_validation, outcomes_test, outcomes_validation = cv.train_test_split(features_intermediate,outcomes_intermediate,test_size=0.5,random_state=0)

print "build model \n"
#modeling.basic_model("LogisticRegression",features_train,outcomes_train,features_test,outcomes_test)
#modeling.basic_model("RandomForestClassifier",features_train,outcomes_train,features_test,outcomes_test)
modeling.basic_model("BernoulliNB",features_train,outcomes_train,features_test,outcomes_test)
# modeling.basic_model("GaussianNB",features_train,outcomes_train,features_test,outcomes_test)
# modeling.basic_model("GradientBoostingClassifier",features_train,outcomes_train,features_test,outcomes_test)
# modeling.basic_model("SVC",features_train,outcomes_train,features_test,outcomes_test) # donot try, takes very very long

# now chose algorithm with the best parameters.
model = LogisticRegression(solver='sag',C=1,n_jobs=-1,random_state=0)
# model = RandomForestClassifier(n_jobs=-1,random_state=0)
# # # model = BernoulliNB(alpha=300)
# # # TODO: model = GradientBoostingClassifier(random_state=0)
# # # TODO: model = SVC(random_state=0)
model.fit(features_train, outcomes_train)

# # make predictions on validation set. use only once to evaluate final model's performance
expected = outcomes_validation
predicted = model.predict(features_validation)
predicted_prob = model.predict_proba(features_validation)

# # summarize the fit of the model
print("log loss: %s \n" % metrics.log_loss(expected, predicted_prob))
print("accuracy score: %s \n" % metrics.accuracy_score(expected, predicted))
print("classification_report: %s \n" % metrics.classification_report(expected, predicted))
print("f1 score: %s \n" % metrics.f1_score(expected, predicted, average='weighted'))
print("confusion matrix: %s \n" % metrics.confusion_matrix(expected, predicted))
