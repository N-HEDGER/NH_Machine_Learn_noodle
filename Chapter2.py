
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

#----- 3) MANIPULATE DATA  -----
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


#----- 4) VISUALIZE DATA  -----

# Create a copy of the housing data 

housing= strat_train_set.copy()

# A combination of plotting options
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
plt.show()


# Look for correlations via .corr() method

corr_matrix=housing.corr()


# Select a subset of attributes and plot them against each other
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

#----- 5) EXPERIMENT WITH ATTRIBUTE COMBINATIONS  -----

# Create new attributes that reflect per house, rather than per district.
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

#----- 5) CLEANING AND PREPARING DATA  -----

# We cant deal with missing features, so we either get rid of the districts,
# the whole attribute, or impute with a median.

from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")

# It will only work on numeric data, so we need to drop ocean_proximity
housing_num = housing.drop('ocean_proximity', axis=1)

# Fit on the numeric data.
imputer.fit(housing_num)

# Check the stats
imputer.statistics_

# Apply the transformation.
X=imputer.transform(housing_num)

# Convert back to a pandas dataframe
housing_tr=pd.DataFrame(X,columns=housing_num.columns)


# Now we want to deal with the text attributes
housing_cat = housing['ocean_proximity']
housing_cat_encoded, housing_categories = housing_cat.factorize()
housing_cat_encoded[:10]


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot


housing_cat_1hot.toarray()

#----- 5) CUSTOM TRANSFORMERS AND PIPELINES  -----

# Class with 3 methods i) fit() ii) transform() iii) fit_transform()
from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

# This is just a function for combining attributes written as a class
# with some other junk I dont really understand
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)


from sklearn.base import BaseEstimator, TransformerMixin


# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# Here we are forming a pipeline of the pre-processing we executed before
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
    ])

# Runs the num_pipeline and the cat_pipeline in paralell, concatenates
# them and returns the output 
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared


#----- 6) TRAINING AND EVALUATING ON THE TRAINING SET  -----




