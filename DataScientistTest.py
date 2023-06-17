import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# load the model

linear_model = LinearRegression()

# load the train and test datasets as DataFrame

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df.head()

train_df.shape

train_df.info()

train_df.describe()

# check if these 3 variables are highly correlated

sns.heatmap(train_df[["surface_area", "agricultural_land", "forest_area"]].corr(), cmap="PRGn", annot=True)
plt.show()

# pairwise relationships and distributions of these variables

sns.pairplot(train_df[["surface_area", "agricultural_land", "forest_area"]])
plt.show()

# create a copy of train_df

train_df2 = train_df.copy()

# having these variable alltogether will not help us for future predictions

train_df2["surface_area"] = train_df2["surface_area"] * 100
train_df2["agricultural_portion"] = train_df2["agricultural_land"] / train_df2["surface_area"]
train_df2["forest_portion"] = train_df2["forest_area"] / train_df2["surface_area"]
train_df2.drop(["agricultural_land", "forest_area"], axis=1, inplace=True)

train_df2.head(10)

# correlation for surface_area and new columns

sns.heatmap(train_df2[["surface_area", "agricultural_portion", "forest_portion"]].corr(), cmap="PRGn", annot=True)
plt.show()

train_df2.isnull().sum()

# there are too many NaN values in inflation_annual, inflation_monthly, inflation_weekly columns
# so fill the inflation_annual column according to the other 2 columns then drop the other 2 columns

for i in train_df2.index:
    if np.isnan(train_df2.loc[i, "inflation_annual"]):
        if np.isnan(train_df2.loc[i, "inflation_monthly"]):
            train_df2.loc[i, "inflation_annual"] = 52 * train_df2.loc[i, "inflation_weekly"]
        else:
            train_df2.loc[i, "inflation_annual"] = 12 * train_df2.loc[i, "inflation_monthly"]

train_df2.drop(["inflation_monthly", "inflation_weekly"], axis=1, inplace=True)

train_df2.isnull().sum()

# see how many are there unique variables in the columns that dtype of it is object

for column_name in train_df2.columns:
    if train_df2[column_name].dtypes == "object":
        unique_amount = len(train_df2[column_name].unique())
        print("The '{}' column has {} unique categories.".format(column_name, unique_amount))
        
# for internet_users there are too many unique variables so create a new column for percent internet users and drop internet_users
        
variables = train_df2["internet_users"].str.split()

train_df2["percent_internet_users"] = (pd.to_numeric(variables.str.get(0), errors="coerce")) / (pd.to_numeric(variables.str.get(2), errors="coerce"))

train_df2.drop(["internet_users"], axis=1, inplace=True)

# convert object variables to numeric for the columns that it's dtype is object

for name in train_df2.select_dtypes(include="object"):
    print(name, ":")
    print(train_df2[name].value_counts(), "\n\n")
    
train_df2["mobile_subscriptions"] = [1 if i == "more than 1 per person" else 2 for i in train_df2["mobile_subscriptions"]]

mapper_1 = {"[0%-25%)": 1, "[25%-50%)": 2, "[50%-75%)": 2, "unknown": 3}
train_df2["women_parliament_seats_rate"].replace(mapper_1, inplace=True)

"""
for old_value, new_value in mapper_1.items():
    if old_value in train_df2["women_parliament_seats_rate"].values:
        train_df2["women_parliament_seats_rate"] = train_df2["women_parliament_seats_rate"].replace(old_value, new_value)
"""
        
"""
train_df2["wome_parliament_seats_rate"] = (train_df2["wome_parliament_seats_rate"].replace("[0%-25%)", 1))
train_df2["wome_parliament_seats_rate"] = (train_df2["wome_parliament_seats_rate"].replace("[0%-25%)", 2))
train_df2["wome_parliament_seats_rate"] = (train_df2["wome_parliament_seats_rate"].replace("[25%-50%)", 2))
train_df2["wome_parliament_seats_rate"] = (train_df2["wome_parliament_seats_rate"].replace("unknown", 3))
"""

mapper_2 = {"very low": 1, "medium low": 2, "low": 3, "medium high": 4, "high": 5, "very high": 6, "unknown": 7}
train_df2["national_income"].replace(mapper_2, inplace=True)

mapper_3 = {"very low access": 1, "low access": 2, "medium access": 3, "high access": 4, "very high access": 5, "no info": 6}
train_df2["improved_sanitation"].replace(mapper_3, inplace=True)

# create a copy of train_df3 and fill the missing values with strategy "median"

train_df3 = train_df2.copy()
train_df3.isnull().sum().sort_values(ascending=False)

imputer = SimpleImputer(strategy="median")
imputer.fit(train_df3)
train_df3 = pd.DataFrame(imputer.transform(train_df3), columns=train_df3.columns)

train_df3.head(10)

# plot histogram for life_expectancy

def plot_histogram(x):
    plt.hist(x, color="gray", edgecolor="black", alpha=0.8)
    plt.title("histogram of {}".format(x.name))
    plt.xlabel("value")
    plt.ylabel("frequency")
    plt.show()
    
plot_histogram(train_df3["life_expectancy"])

test_df.head()

# create a copy of test_df do the same things you do for train_df to test_df2

test_df2 = test_df.copy()

test_df2["surface_area"] = test_df2["surface_area"] * 100
test_df2["agricultural_portion"] = test_df2["agricultural_land"] / test_df2["surface_area"]
test_df2["forest_portion"] = test_df2["forest_area"] / test_df2["surface_area"]
test_df2.drop(["agricultural_land", "forest_area"], axis=1, inplace=True)

for i in test_df2.index:
    if np.isnan(test_df2.loc[i, "inflation_annual"]):
        if np.isnan(test_df2.loc[i, "inflation_monthly"]):
            test_df2.loc[i, "inflation_annual"] = 52 * test_df2.loc[i, "inflation_weekly"]
        else:
            test_df2.loc[i, "inflation_annual"] = 12 * test_df2.loc[i, "inflation_monthly"]

test_df2.drop(["inflation_monthly", "inflation_weekly"], axis=1, inplace=True)

variables2 = test_df2["internet_users"].str.split()

test_df2["percent_internet_users"] = (pd.to_numeric(variables2.str.get(0), errors="coerce")) / (pd.to_numeric(variables2.str.get(2), errors="coerce"))

test_df2.drop(["internet_users"], axis=1, inplace=True)

test_df2["mobile_subscriptions"] = [1 if i == "more than 1 per person" else 2 for i in test_df2["mobile_subscriptions"]]

test_df2["women_parliament_seats_rate"].replace(mapper_1, inplace=True)

test_df2["national_income"].replace(mapper_2, inplace=True)

test_df2["improved_sanitation"].replace(mapper_3, inplace=True)

imputer.fit(test_df2)
test_df2 = pd.DataFrame(imputer.transform(test_df2), columns=test_df2.columns)

# split train_df3 for x axis and y axis

y = train_df3["life_expectancy"]
X = train_df3.drop(["life_expectancy"], axis=1)

y.head()
X.head()

# train the model with X and y

linear_model.fit(X, y)

# make predictions for features in the train_df3

y_pred = linear_model.predict(X)

# plot historgram for the comparison of true "life_expectancy" and predicted "life_expectancy" values

def plot_hist_comp(x, y):
    plt.hist(x, alpha=0.5, edgecolor="black", label="Actual")
    plt.hist(y, alpha=0.5, edgecolor="black", label="Predicted")
    plt.title("Histogram of actual outcomes v.s predicted outcomes")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend(loc="upper left")
    plt.show()
    
plot_hist_comp(y, y_pred)

# print mean abosulte error

mae = mean_absolute_error(y, y_pred)
mae

# make predictions for unseen data and save is as .csv file

unseen_test = linear_model.predict(test_df2)
d = {"": test_df.iloc[:, 0], "life_expectancy": unseen_test}
submission = pd.DataFrame(data=d)
submission.to_csv("submission.csv", index=False)

