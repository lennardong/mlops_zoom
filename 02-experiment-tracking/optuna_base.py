"""
In this rewritten code:

We import the necessary libraries, including optuna instead of hyperopt.
We set the MLflow tracking URI and experiment name to "random-forest-optuna".
The load_pickle function remains the same.
In the run_optimization function:

We define an objective function that takes an Optuna trial object as input.
Inside the objective function, we use the trial.suggest_* methods to define the search space for each hyperparameter.
We create an MLflow run and train the Random Forest model with the sampled hyperparameters.
We evaluate the model on the validation set and calculate the RMSE.
We log the hyperparameters and RMSE to MLflow.
Finally, we return the RMSE as the objective value to be minimized.


We create an Optuna study with the direction set to "minimize" the objective value.
We optimize the study by calling study.optimize with the objective function and the number of trials.
After the optimization is complete, we print the best hyperparameters found by Optuna.

The main differences compared to the Hyperopt version are:

We use Optuna's trial.suggest_* methods to define the search space instead of Hyperopt's hp functions.
We create an Optuna study and optimize it using study.optimize instead of using Hyperopt's fmin function.
We don't need to specify the search algorithm explicitly as Optuna automatically selects an appropriate one.

The rest of the code structure remains similar, with the objective function being called by the optimizer to evaluate different hyperparameter configurations and log the results to MLflow.
"""

import os
import pickle

import click
import mlflow
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("random-forest-optuna")


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved",
)
@click.option(
    "--num_trials",
    default=15,
    help="The number of parameter evaluations for the optimizer to explore",
)
def run_optimization(data_path: str, num_trials: int):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 1, 20),
            "n_estimators": trial.suggest_int("n_estimators", 10, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "random_state": 42,
        }

        with mlflow.start_run():
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_params(params)
            mlflow.log_metric("rmse", rmse)

        return rmse

    # study = optuna.create_study(direction="minimize")
    # study.optimize(objective, n_trials=num_trials)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=num_trials)
    best_params = study.best_params
    print("Best hyperparameters:", best_params)


if __name__ == "__main__":
    run_optimization()
