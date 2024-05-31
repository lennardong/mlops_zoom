import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from IPython.display import display, HTML

def load_data(filename: str):
    df = pd.read_parquet(filename)
    df["duration"] = (df.lpep_dropoff_datetime - df.lpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    return df

def preprocess_features(df: pd.DataFrame, categorical: list, numerical: list):
    df[categorical] = df[categorical].astype(str)
    train_dicts = df[categorical + numerical].to_dict(orient="records")
    return train_dicts

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def plot_predictions(y_train, y_pred):
    sns.histplot(y_pred, label='Prediction', color='red', element='step', fill=True, stat='density')
    sns.histplot(y_train, label='Actual', color='grey', element='step', fill=True, stat='density')
    plt.legend()
    plt.xlabel("Duration")
    plt.ylabel("%")
    plt.show()

    
if __name__ == "__main__":
    df = load_data("../_data/green_tripdata_2024-01.parquet")
    categorical = ["PULocationID", "DOLocationID"]
    numerical = ["trip_distance"]
    
    train_dicts = preprocess_features(df, categorical, numerical)
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    
    y_train = df['duration'].values
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_train)
    
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"Root Mean Squared Error: {mse:.2f}")
    
    plot_predictions(y_train, y_pred)
