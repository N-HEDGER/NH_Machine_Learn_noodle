
#----- 1) IMPORTS  -----

# Import libraries
import os
import tarfile
from six.moves import urllib


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# Function for collecting data.
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
	# If the directory doesnt exist
    if not os.path.isdir(housing_path):
    	#Make it
        os.makedirs(housing_path)
        # Get the file and exract it.
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# Pandas is a library for creating nice data structures
import pandas as pd

# Use pandas function to read CSV (default) into what is basically
# an R dataframe.

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()


#----- 2) INSPECT DATA  -----

# Head shows first N rows of dataframe.
housing.head()

# Housing is a pandas dataframe with 10 columns and 20640 rows
type(housing)
housing.shape
# Returns the variable types.
housing.info()

# Return different attributes - (like the $ operator in R)
housing["ocean_proximity"]

# (Like the table function in R)
housing["ocean_proximity"].value_counts()

# Provides a table of the numeric variables.
housing.describe()

%matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()






