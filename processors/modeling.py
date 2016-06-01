import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn import grid_search as gs

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
        #param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        param_grid = {'C': [ 1]}

    cv_model = gs.GridSearchCV(algo, param_grid, cv=folds, scoring='accuracy')
    cv_model.fit(trainingset_features, trainingset_outcomes)
    print "scores for each model %s" % cv_model.grid_scores_
    # examine the best model
    print "best parameter value %s" % cv_model.best_params_
    print "score for best model %s" % cv_model.best_score_

    return cv_model.best_params_
