from pyzipcode import ZipCodeDatabase

zcdb = ZipCodeDatabase()
sf_zipcodes = zcdb.find_zip(city="San Francisco", state="CA")

# function to convert longitude and latitude into zip code