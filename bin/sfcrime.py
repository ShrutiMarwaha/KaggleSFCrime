import pandas as pd

# local functions/processors
import processors.loader as loader
import processors.feature_extractor as extractor

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

training_striped_time = extractor.extract_date_dataframe(training_set.Dates)
test_striped_time = extractor.extract_date_dataframe(test_set.Dates)

training_features = pd.concat( (training_striped_time,training_set[["DayOfWeek","PdDistrict","X","Y","Category"]]), axis=1)
print "after features extraction - training data: \n %s \n" % training_features.head()
test_features = pd.concat( (test_striped_time,test_set[["DayOfWeek","PdDistrict","X","Y"]]), axis=1)
print "after features extraction - training datatest data: \n %s \n" % test_features.head()


