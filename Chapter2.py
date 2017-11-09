
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

# Plots all numeric data.
housing.hist(bins=50, figsize=(20,15))
plt.show()

#----- 2) MANIPULATE DATA  -----
# Set seed so that all output can be replicated
import numpy as np
np.random.seed(42)
from sklearn.model_selection import train_test_split

# Use sklearn function to split data into a training and test set
# 80% train 20% test.
train_set, test_set = train_test_split(housing,test_size=0.2,random_state=42)


# Divide median income by 1.5 to limit the number of income categories
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
# Label those above 5 as 5
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

# Now we have 5 categories
housing["income_cat"].value_counts()

# Now we do some stratified samping so that we have a representative number of
# cases from each category

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Now drop income category, because we have done our stratified sampling
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

