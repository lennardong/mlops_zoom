import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error

from IPython.display import display, HTML

####################

df = pd.read_parquet("../_data/green_tripdata_2024-01.parquet")

# duration in minutes, as float
# filter only those within 1 hour
df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
df = df[(df.duration >= 1) & (df.duration <= 60)]

# Set Y parameters
categorical = ["PULocationID", "DOLocationID"]
numerical = ["trip_distance"]
df[categorical] = df[categorical].astype(str)  # ? what is this doing, why not use a label or onehot encoder?

train_dicts = df[categorical + numerical].to_dict(orient="records")
dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)

target = 'duration'
y_train = df[target].values


# ? why is X uppercase but y lowercase?
# ? what other terms are used to refer to X and y in statistics? is it dep and indep values? 

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_train)

mse = mean_squared_error(y_train, y_pred, squared=False)
print(mse)

sns.histplot(y_pred, label='prediction')
sns.histplot(y_train, label='actual')

# ? fix so that it is overlayed on one chart. intent is to use distplot, I want a light fill with line
plt.legend()

if __name__ == "__main__":
    display(df)