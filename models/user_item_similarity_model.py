import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, normalize


def train_item_based_recommendation_model(
    data_df: pd.DataFrame,
    user_profiles: pd.DataFrame,
    all_item_profiles: pd.DataFrame,
    user_encoder: LabelEncoder,
    item_encoder: LabelEncoder,
    calc_metrics: bool = False,
) -> np.ndarray:
    """Calculate the simularity matrix for the item based recommendation model."""

    _user_profiles = user_profiles.copy()
    _all_item_profiles = all_item_profiles.copy()

    # Sort profiles DataFrames on their encoded label and reset index
    _user_profiles["user_objectid"] = user_encoder.transform(
        _user_profiles["user_objectid"]
    )
    _all_item_profiles["item_objectid"] = item_encoder.transform(
        _all_item_profiles["item_objectid"]
    )
    _user_profiles = _user_profiles.sort_values(by="user_objectid").reset_index(
        drop=True
    )
    _all_item_profiles = _all_item_profiles.sort_values(by="item_objectid").reset_index(
        drop=True
    )

    # Ensure both profiles DataFrames have the same naming convention for columns
    new_user_columns = [
        col.replace("user_", "") if col != "user_objectid" else "user_objectid"
        for col in _user_profiles.columns
    ]
    new_item_columns = [
        col.replace("item_", "") if col != "item_objectid" else "item_objectid"
        for col in _all_item_profiles.columns
    ]
    _user_profiles.columns = new_user_columns
    _all_item_profiles.columns = new_item_columns

    _user_profiles = _user_profiles.drop(columns="user_objectid")
    _all_item_profiles = _all_item_profiles.drop(columns="item_objectid")

    # Normalize the profiles
    user_profiles_norm = normalize(_user_profiles, axis=1)
    item_profiles_norm = normalize(_all_item_profiles, axis=1)

    # Calculate similarity (dot product between user and item profiles)
    similarity_matrix = np.dot(user_profiles_norm, item_profiles_norm.T)

    user_encoded_labels = data_df["user_objectid"].values
    item_encoded_labels = data_df["item_objectid"].values
    y_true = data_df["label"].values
    y_pred = similarity_matrix[user_encoded_labels, item_encoded_labels]

    if calc_metrics:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        print("Model Performance Metrics:")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
    return similarity_matrix

