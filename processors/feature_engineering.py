from pyzipcode import ZipCodeDatabase

zcdb = ZipCodeDatabase()
sf_zipcodes = zcdb.find_zip(city="San Francisco", state="CA")