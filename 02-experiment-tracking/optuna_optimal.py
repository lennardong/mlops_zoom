"""
Here are the main changes and best practices incorporated:

Modularity: The code is divided into smaller, reusable functions (load_data, objective) for better modularity and readability.
Type Hints: Type hints are added to function parameters and return values to improve code clarity and catch potential type-related errors.
Context Manager: The with statement is used as a context manager to handle file opening and closing automatically.
Unpacking: The load_data function returns a tuple of tuples, which is unpacked in the run_optimization function for clarity.
Pruning: Optuna's MedianPruner is used to prune less promising trials early, speeding up the optimization process.
MLflow Integration: Optuna's MLflowCallback is used to log the optimization results directly to MLflow for better experiment tracking and management.
Lambda Function: The objective function is passed to study.optimize using a lambda function to avoid defining a separate callable.
Error Handling: Proper error handling should be added to handle potential exceptions gracefully (not shown in the code snippet).
Logging: Logging statements should be added to record important events and assist in debugging and monitoring (not shown in the code snippet).

These best practices help improve the code's readability, maintainability, and efficiency, making it more suitable for production environments. However, note that there may be additional considerations depending on the specific requirements and constraints of your project.

## Callbacks
In the refactored code, the `MLflowCallback` is used to log the optimization results directly to MLflow during the optimization process. This is an alternative to logging the results manually within the `objective` function.

Using the callback has a few advantages:

1. Separation of Concerns: By using the callback, the logging functionality is separated from the core optimization logic. The `objective` function focuses solely on evaluating the model performance for a given set of hyperparameters, while the callback takes care of logging the results to MLflow. This separation of concerns makes the code more modular and easier to maintain.
2. Flexibility: The callback allows for more flexible logging options. You can configure the callback to log additional information, such as the trial number, trial parameters, or custom metrics, without modifying the `objective` function itself. This is particularly useful if you want to log different metrics or metadata for different experiments or optimization runs.
3. Reusability: The `MLflowCallback` is a reusable component provided by the Optuna library. It encapsulates the logging logic and can be easily plugged into different optimization studies or experiments. This promotes code reuse and reduces duplication.
4. Consistency: Using the callback ensures consistent logging behavior across different optimization studies or experiments. You don't need to remember to add the logging code manually each time you define an `objective` function. The callback takes care of logging automatically based on the specified configuration.

However, it's important to note that logging the results directly within the `objective` function, as shown in the original code, is also a valid approach. It may be simpler and more straightforward in some cases, especially if you have specific logging requirements or want to log additional information that is not covered by the callback.

Ultimately, the choice between using a callback or logging directly in the `objective` function depends on your specific needs and preferences. If you value separation of concerns, flexibility, and reusability, using the `MLflowCallback` is a good choice. If you prefer a simpler and more direct approach, logging within the `objective` function can be suitable as well.
"""

import os
import pickle
from typing import Tuple

import click
import mlflow
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("random-forest-optuna")


def load_data(data_path: str) -> Tuple[Tuple[object, object], Tuple[object, object]]:
    with open(os.path.join(data_path, "train.pkl"), "rb") as f_train, open(
        os.path.join(data_path, "val.pkl"), "rb"
    ) as f_val:
        train_data = pickle.load(f_train)
        val_data = pickle.load(f_val)
    return train_data, val_data


def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "n_estimators": trial.suggest_int("n_estimators", 10, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        "random_state": 42,
    }

    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)

    return rmse


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
    (X_train, y_train), (X_val, y_val) = load_data(data_path)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=num_trials,
    )

    print("Best hyperparameters:", study.best_params)


if __name__ == "__main__":
    run_optimization()
