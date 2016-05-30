# club different crimes into different classes of crimes. http://criminal.findlaw.com/criminal-law-basics/classifications-of-crimes.html
crime_dict ={"infraction":["LOITERING"], "misdemeanor":["BAD CHECKS", "BRIBERY", "DISORDERLY CONDUCT","DRUG/NARCOTIC", "DRIVING UNDER THE INFLUENCE", "DRUNKENNESS", "EMBEZZLEMENT", "FAMILY OFFENSES",
                                                        "FORGERY/COUNTERFEITING", "GAMBLING", "LIQUOR LAWS", "MISSING PERSON", "NON-CRIMINAL", "OTHER OFFENSES", "PORNOGRAPHY/OBSCENE MAT",
                                                        "PROSTITUTION", "RECOVERED VEHICLE", "RUNAWAY", "SECONDARY CODES", "STOLEN PROPERTY", "SUICIDE", "SUSPICIOUS OCC", "TREA", "TRESPASS", "WARRANTS"],
             "felony": ["ARSON", "ASSAULT", "BURGLARY", "EXTORTION", "FRAUD", "KIDNAPPING", "LARCENY/THEFT", "ROBBERY", "SEX OFFENSES FORCIBLE", "SEX OFFENSES NON FORCIBLE", "VANDALISM", "VEHICLE THEFT", "WEAPON LAWS"]}

def get_key(dict_name,value_name):
    '''
    from a dictionary, given a value, get the key

    :param dict_name: dictionary
    :param value_name: input value
    :return k: key associated with input value
    '''
    for k, v in dict_name.iteritems():
        if value_name in v:
             return k

from processors import loader
import pandas as pd

# training_set = loader.load_csv_data("/Users/shruti/Desktop/WorkMiscellaneous/MachineLearning/SanFranciscoCrime/train.csv",n_rows=100)
#
# for row in training_set.itertuples():
#     training_set.loc[row.Index,"crime_class"] = get_key(crime_dict,row.Category)



