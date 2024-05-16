from typing import List, Optional, Union
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix


class TaxiTripPredictor:
    def __init__(self, filename: str) -> None:
        self.filename: str = filename
        self.model: Union[LinearRegression, Ridge] = None  # type: ignore
        self.vectorizer: DictVectorizer = DictVectorizer()
        self.scaler: StandardScaler = None  # type: ignore
        self.df: pd.DataFrame = None  # type: ignore
        self.X_train: csr_matrix = None  # type: ignore
        self.y_train: pd.Series = None  # type: ignore

    def load_and_preprocess_data(
        self, categorical: List[str], numerical: List[str], use_scaling: bool = False
    ) -> None:
        try:
            self.df = pd.read_parquet(self.filename)
            self.df["duration"] = (
                self.df.lpep_dropoff_datetime - self.df.lpep_pickup_datetime
            ).dt.total_seconds() / 60
            self.df = self.df[(self.df.duration >= 1) & (self.df.duration <= 60)]
            self.df[categorical] = self.df[categorical].astype(str)
            train_dicts = self.df[categorical + numerical].to_dict(orient="records")
            self.X_train = self.vectorizer.fit_transform(train_dicts)
            if use_scaling:
                self.scaler = StandardScaler(with_mean=False)
                self.X_train = self.scaler.fit_transform(self.X_train)
            self.y_train = self.df["duration"].values
        except Exception as e:
            print(f"Failed to load or preprocess data: {e}")

    def train_model(self, use_ridge: bool = False) -> None:
        if use_ridge:
            self.model = Ridge(alpha=1.0)
        else:
            self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

    def predict_and_evaluate(self) -> None:
        try:
            y_pred = self.model.predict(self.X_train)
            mse = mean_squared_error(self.y_train, y_pred, squared=False)
            print(f"Root Mean Squared Error: {mse:.2f}")
            self.plot_predictions(y_pred)
        except Exception as e:
            print(f"Error during prediction or evaluation: {e}")

    def plot_predictions(self, y_pred: pd.Series) -> None:
        plt.figure(figsize=(10, 6))
        sns.histplot(self.y_train, label="Actual", color="grey", element="step", fill=True, stat="density")  # type: ignore
        sns.histplot(y_pred, label="Prediction", color="blue", element="step", fill=True, stat="density")  # type: ignore
        plt.title("Comparison of Predictions and Actual Durations")
        plt.xlabel("Duration (minutes)")
        plt.ylabel("Density")
        plt.legend()
        plt.show()


def main() -> None:
    predictor = TaxiTripPredictor("../_data/green_tripdata_2024-01.parquet")
    predictor.load_and_preprocess_data(
        ["PULocationID", "DOLocationID"], ["trip_distance"], use_scaling=False
    )
    predictor.train_model(use_ridge=False)
    print("Experiment 1 (No Scaling, No Ridge):")
    predictor.predict_and_evaluate()

    predictor.load_and_preprocess_data(
        ["PULocationID", "DOLocationID"], ["trip_distance"], use_scaling=True
    )
    predictor.train_model(use_ridge=True)
    print("Experiment 2 (With Scaling, With Ridge):")
    predictor.predict_and_evaluate()

    predictor.load_and_preprocess_data(
        ["PULocationID", "DOLocationID"], ["trip_distance"], use_scaling=True
    )
    predictor.train_model(use_ridge=False)
    print("Experiment 2 (With Scaling, Without Ridge):")
    predictor.predict_and_evaluate()


if __name__ == "__main__":
    main()
