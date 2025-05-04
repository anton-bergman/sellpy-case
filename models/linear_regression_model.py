import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_linear_regression_model(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    save_model_to: str | None = None,
) -> LinearRegression:
    """Trains a linear regression model on the training data and evaluates it on the test data.

    Args:
        train (pd.DataFrame): Training data containing features and labels.
        test (pd.DataFrame, optional): Test data containing features and labels.
        save_model_to (str, optional): Path to save the trained model. Defaults to None. If None, the model is not saved.

    Returns:
        LinearRegression: The trained linear regression model.
    """
    X_train = train.drop(columns=["label"])
    y_train = train["label"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    if test is not None:
        X_test = test.drop(columns=["label"])
        y_test = test["label"]

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print("Model Performance Metrics:")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")

    if save_model_to is not None:
        with open(save_model_to, "wb") as file:
            pickle.dump(model, file)
        print(f"Model saved as '{save_model_to}'")

    return model


def load_linear_regression_model(
    filename: str, test: pd.DataFrame | None = None
) -> LinearRegression:
    """Loads a linear regression model from a file and evaluates it on the test data."""
    with open(filename, "rb") as file:
        model: LinearRegression = pickle.load(file)
    print(f"Model loaded from '{filename}'")

    if test is not None:
        X_test = test.drop(columns=["label"])
        y_test = test["label"]
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print("Model Performance Metrics:")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")

    return model

