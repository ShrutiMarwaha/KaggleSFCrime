from pyzipcode import ZipCodeDatabase
from scipy.spatial import distance
import pandas as pd

zcdb = ZipCodeDatabase()
sf_zipcodes = zcdb.find_zip(city="San Francisco", state="CA")
print(len(sf_zipcodes))

# extract longitude, latitude and zip codes from sf_zipcodes
sf_zipcodes_df = pd.DataFrame()

for sf_zipcode in sf_zipcodes:
    temp_dict = {'latitude':sf_zipcode.latitude, 'longitude':sf_zipcode.longitude, 'zipcode':int(sf_zipcode.zip)}
    temp_df = pd.DataFrame(temp_dict, index=[0])
    sf_zipcodes_df = pd.concat([sf_zipcodes_df, temp_df])

print(sf_zipcodes_df.shape)
print(sf_zipcodes_df.head())

X=-122.425
Y=37.774

for sf_zipcode in sf_zipcodes:
    lat = sf_zipcode.latitude
    long = sf_zipcode.longitude
    euclidean_dist = round( distance.euclidean((long,lat) , (X,Y)), 4)
    print(euclidean_dist)
    #print(sf_zipcode.zip)





zips = [sf_zipcode.zip for sf_zipcode in sf_zipcodes]
# function to convert longitude and latitude into zip code