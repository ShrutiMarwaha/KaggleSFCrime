from pyzipcode import ZipCodeDatabase
import pandas as pd

# this file is only for test and trial. please ignore

###### feature_engineering.py
zcdb = ZipCodeDatabase()
sf_zipcodes = zcdb.find_zip(city="San Francisco", state="CA")


# extract longitude, latitude and zip codes from sf_zipcodes
sf_zipcodes_df = pd.DataFrame()

for sf_zipcode in sf_zipcodes:
    temp_dict = {'latitude':sf_zipcode.latitude, 'longitude':sf_zipcode.longitude, 'zipcode':int(sf_zipcode.zip)}
    temp_df = pd.DataFrame(temp_dict, index=[0])
    sf_zipcodes_df = pd.concat([sf_zipcodes_df, temp_df])

print(sf_zipcodes_df.shape)
print(sf_zipcodes_df.head())

############# sfcrime.py

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
print(metrics.accuracy_score(expected, predicted))
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
# print(metrics.roc_auc_score(expected, predicted)) # predicted outputs have to be binarized
#############################



# divide data in to training and validation set. Since performing cross validation, no need for test set
# validation set will be only used once to evaluate final model's performance
cv_features_train, cv_features_validation, cv_outcomes_train, cv_outcomes_validation = cv.train_test_split(training_dummy_var2,outcomes2,test_size=0.4,random_state=0)

# divide data in to intermediate set into test and validation set. validation set will be only used once to evalaute final model's performance
# cv_features_test, cv_features_validation, cv_outcomes_test, cv_outcomes_validation = cv.train_test_split(cv_features_intermediate,cv_outcomes_intermediate,test_size=0.5,random_state=0)
print(cv_features_train.shape)
print(cv_features_validation.shape)

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

# GridSearch with 10 fold CV will take very long, so running only 3 folds to find the best parameters.
# use RandomizedSearchCV searches a subset of the parameters to reduce computational expense
model = gs.GridSearchCV(algo, param_grid, cv=3, scoring='accuracy')
# TODO: model = gs.GridSearchCV(algo, param_grid, cv=3, scoring='f1_samples',n_jobs=-1)
# scoring: f1_samples (for multilabel sample); f1 (for for binary targets); accuracy (for model accuracy)
model.fit(cv_features_train, cv_outcomes_train)

# model scores for each parameter used in grid
print model.grid_scores_
# examine the best model
print model.best_score_
print model.best_params_
# print model.best_estimator_

# now chose algorithm with the best parameters.
# this step can be avoided if all desired arguments are used in GridSearchCV. GridSearchCV automatically refits the best model.
algo = LogisticRegression(solver='lbfgs',multi_class='multinomial',C=1,n_jobs=-1,random_state=0)
# algo = RandomForestClassifier(n_estimators=200,max_depth=15,n_jobs=-1,random_state=0)
# algo = BernoulliNB(alpha=300)
# TODO: algo = GradientBoostingClassifier(random_state=0)

# apply 10 fold cross validation.
model = cv.cross_val_score(estimator=algo, X=cv_features_train, y=cv_outcomes_train, cv=10, scoring='accuracy')
# TODO: model = cv.cross_val_score(estimator=algo, X=cv_features_train, y=cv_outcomes_train, cv=10, scoring='f1_samples')

# use average accuracy as an estimate of out-of-sample accuracy
print model.mean()

# make predictions on validation set. use only once to evaluate final model's performance
predicted = model.predict(cv_features_validation)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
