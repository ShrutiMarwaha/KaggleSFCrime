from pyzipcode import ZipCodeDatabase
from scipy.spatial import distance

zcdb = ZipCodeDatabase()
sf_zipcodes = zcdb.find_zip(city="San Francisco", state="CA")
# print(len(sf_zipcodes))

def closest_zipcode(input_longitude,input_latitude):
    """
    function to convert latitude and longitude to zipcode:
    find euclidean distance between user provided latitude and longitude and
    all latitudes, longitudes from sf_zipcodes and select the row from latter
    which has minimum distance. then extract its zip code

    :param input_longitude,input_latitude: latitude and longitude that you want to convert to closest zipcode
    :return closest_zip: zip code
    """

    closest_zip = None
    euc_dist = None

    for i, sf_zipcode in enumerate(sf_zipcodes):
        # extract latitude and longitude from each row in sf_zipcode
        lat = sf_zipcode.latitude
        long = sf_zipcode.longitude
        # calculate euclidean distance between lat and long from sf_zipcode and input latitude and longitude value
        euclidean_dist = round( distance.euclidean((long,lat), (input_longitude,input_latitude)), 4)
        # assign the euclidean distance calculated for first row as euc_dist
        if i == 0:
            euc_dist = euclidean_dist

        # if the euclidean_dist is smaller than the one calculated in previous iteration,
        # replace it with current one and extract the corresponding zipcode
        if euclidean_dist < euc_dist:
            euc_dist = euclidean_dist
            closest_zip = int(sf_zipcode.zip)

    return closest_zip

# closest_zipcode(-122.4033, 37.71343)
