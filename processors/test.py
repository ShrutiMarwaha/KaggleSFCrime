from pyzipcode import ZipCodeDatabase
import pandas as pd

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