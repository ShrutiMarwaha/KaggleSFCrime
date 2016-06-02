# import local functions/processors
from processors import loader
import analysis
from processors import feature_extractor as extractor
from processors import feature_engineering as engineering
from processors import modeling
# import libraries
import pandas as pd
from sklearn import cross_validation as cv
from sklearn import metrics
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
# save file for future use
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
# remove the columns that should not be used for model building
training_features = training_set.drop(["Category","Dates","Resolution","Descript","Address","X","Y"], axis=1)
print "training set:\n%s" % training_features.head(3)
test_features = test_set.drop(["Dates","Address","X","Y"], axis=1)

# save files for future use
# training_features.to_pickle("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/training_features.pkl")
# test_features.to_pickle("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/test_features.pkl")

# directly load following files and skip all above commands than import libraries
# training_features = pd.read_pickle("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/training_features.pkl")
# outcomes = outcomes = pd.read_pickle("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/outcomes.pkl")
# test_features = pd.read_pickle("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/test_features.pkl")

# Create Dummy Variables from Categorical Data
train_categorical_columns = list(training_features)
print train_categorical_columns
# decide which columns should be categorical and converted to dummy variables. this step cannot be automated, pay attention !!
training_dummy_var = modeling.create_dummy_var(training_features,train_categorical_columns)
analysis.data_summary(training_dummy_var)

test_categorical_columns = list(test_features)
print test_categorical_columns
#do not include first column - ID
test_categorical_columns.remove("Id")
test_dummy_var = modeling.create_dummy_var(test_features.drop(["Id"], axis=1),test_categorical_columns)
analysis.data_summary(test_dummy_var)


###################### Grid Search with Cross Validation ######################
# to run k-fold cross validation, remove classes which have less than "k" samples.
# cv< minimum no.of samples in each class. So either remove such samples or use KFold
# outcomes_frequency = outcomes.value_counts(ascending=True)
outcomes_frequency.head()
outcomes_frequency[outcomes_frequency < 10 ]
matchings_indices = [ i for i, value in enumerate(outcomes) if value == "TREA" ]
outcomes[matchings_indices]
# remove these samples from features and outcomes data
outcomes2 = outcomes.drop(matchings_indices)
training_dummy_var2 = training_dummy_var.drop(matchings_indices)

# divide data in to training and validation set. Since performing cross validation, no need for test set
# validation set will be only used once to evaluate final model's performance
cv_features_train, cv_features_validation, cv_outcomes_train, cv_outcomes_validation = cv.train_test_split(training_dummy_var2,outcomes2,test_size=0.4,random_state=0)
print(cv_features_train.shape)
print(cv_features_validation.shape)

# GridSearch with 10 fold CV will take very long, so running only 2 folds to find the best parameters.
# use RandomizedSearchCV that searches a subset of the parameters to reduce computational expense
# Look at modeling.py for parameters to vary in grid search
modeling.gridsearch_cv_model("LogisticRegression",2,cv_features_train,cv_outcomes_train)
modeling.gridsearch_cv_model("RandomForestClassifier",2,cv_features_train,cv_outcomes_train)
modeling.gridsearch_cv_model("BernoulliNB",2,cv_features_train,cv_outcomes_train)
# modeling.gridsearch_cv_model("GradientBoostingClassifier",2,cv_features_train,cv_outcomes_train) #takes very very long

# now chose algorithm with the best parameters. And apply 10 fold Cross Validation.
# this step can be avoided if all desired arguments and k-fold cv are used in GridSearchCV. GridSearchCV automatically refits the best model.
algo=LogisticRegression(solver='lbfgs',multi_class='multinomial',C=1,n_jobs=-1,random_state=0)
# algo = RandomForestClassifier(n_jobs=-1,random_state=0)
# algo = BernoulliNB(alpha=300)
# algo = GradientBoostingClassifier(random_state=0)
cvmodel_accuracy = cv.cross_val_score(estimator=algo, X=cv_features_train, y=cv_outcomes_train, cv=10, scoring='accuracy')
print "average accuracy score for cross validation %s: \n" % cvmodel_accuracy.mean()
# use 'f1_weighted' as scoring metric because this data has unbalanced classes
cvmodel_f1 = cv.cross_val_score(estimator=algo, X=cv_features_train, y=cv_outcomes_train, cv=10, scoring='f1_weighted')
print "average f1 score for cross validation %s: \n" % cvmodel_f1.mean()
cvmodel_logloss = cv.cross_val_score(estimator=algo, X=cv_features_train, y=cv_outcomes_train, cv=10, scoring='log_loss')
print "average log-loss score for cross validation %s: \n" % cvmodel_logloss.mean()

# now chose algorithm with the best parameters.
model = RandomForestClassifier(n_jobs=-1,random_state=0)
# model = LogisticRegression(solver='lbfgs',multi_class='multinomial',C=1,n_jobs=-1,random_state=0)
# model = BernoulliNB(alpha=300)
# model = GradientBoostingClassifier(random_state=0)
model.fit(cv_features_train, cv_outcomes_train)

# make predictions on validation set. use it only once to evaluate final model's performance
expected = cv_outcomes_validation
predicted = model.predict(cv_features_validation)
predicted_prob = model.predict_proba(cv_features_validation)

# summarize the fit of the model on validation set
print("accuracy score: %s \n" % metrics.accuracy_score(expected, predicted))
print("f1 score: %s \n" % metrics.f1_score(expected, predicted, average='weighted'))
print("log loss: %s \n" % metrics.log_loss(expected, predicted_prob))
print("classification_report: %s \n" % metrics.classification_report(expected, predicted))
#print("confusion matrix: %s \n" % metrics.confusion_matrix(expected, predicted))

############################### final model for submission to kaggle #############################
final_model = LogisticRegression(solver='lbfgs',multi_class='multinomial',C=1,n_jobs=-1,random_state=0)
final_model.fit(training_dummy_var, outcomes)

# make predictions
predicted_prob = final_model.predict_proba(test_dummy_var)
result = pd.DataFrame(predicted_prob, columns=final_model.classes_)
final_result = pd.concat([ test_set.Id,result ], axis=1)

pd.set_option('display.max_columns', None)
print final_result.head()
print final_result.shape

final_result.to_csv("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/Datasets/final_result.csv",index=False)
