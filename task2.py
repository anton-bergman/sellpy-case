import random
from enum import Enum

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from models.linear_regression_model import (
    load_linear_regression_model,
    train_linear_regression_model,
)
from models.mlp import load_MLP_model, train_MLP_model
from models.user_item_similarity_model import train_item_based_recommendation_model
from utils.evaluate import evaluate_user_recommendations
from utils.preprocess import (
    get_dataset,
    get_preprocessed_data,
    get_subset_by_column_substring,
    read_csv_data,
)


class ModelType(Enum):
    LINEAR_REGRESSION = "LinearRegression"
    USER_ITEM_SIMILARITY = "UserItemSimilarity"
    MLP = "MLP"


class Mode(Enum):
    # Train the model on the full user_item_interactions.csv dataset
    TRAIN_ON_FULL_DATA = "train_on_full_data"

    # Train the model on the split user_item_interactions.csv dataset
    TRAIN_ON_SPLIT_DATA = "train_on_split_data"

    # Load the model trained on the full dataset and evaluate it on the live_items_data.csv
    LOAD_MODEL_FOR_FULL_DATA = "load_model_for_full_data"

    # Load the model trained on the split dataset and evaluate it on the test split
    LOAD_MODEL_FOR_SPLIT_DATA = "load_model_for_split_data"


def main() -> None:
    # Hyperparameters
    # ===============
    SEED = 12345
    MODEL = ModelType.USER_ITEM_SIMILARITY
    MODE = Mode.LOAD_MODEL_FOR_FULL_DATA
    TOP_N_RECOMMENDATIONS = 10
    USER_OBJECTID = "00k2cfIsKw"
    # ===============

    # Set seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load Data
    user_features_path = "data/user_features.csv"
    live_item_data_path = "data/live_item_data.csv"
    user_item_interactions_path = "data/user_item_interactions.csv"
    item_data_for_interactions_path = "data/item_data_for_interactions.csv"
    user_features_df = read_csv_data(user_features_path)
    live_item_data_df = read_csv_data(live_item_data_path)
    user_item_interactions_df = read_csv_data(user_item_interactions_path)
    item_data_for_interactions_df = read_csv_data(item_data_for_interactions_path)

    # Preprocess data: Build profiles for users and items
    user_profiles, item_profiles, live_item_profiles, all_item_profiles = (
        get_preprocessed_data(
            user_features_df, item_data_for_interactions_df, live_item_data_df
        )
    )

    # Create dataset
    data_df, user_encoder, item_encoder = get_dataset(
        user_features_df,
        user_item_interactions_df,
        item_data_for_interactions_df,
        live_item_data_df,
        user_profiles,
        item_profiles,
        live_item_profiles,
        seed=SEED,
    )
    split_data: list[pd.DataFrame] = train_test_split(
        data_df, test_size=0.2, random_state=SEED
    )
    train_df, test_df = (
        split_data[0].reset_index(drop=True),
        split_data[1].reset_index(drop=True),
    )

    # Model Training
    if MODEL == ModelType.MLP:
        if MODE == Mode.TRAIN_ON_FULL_DATA:
            model = train_MLP_model(
                train_df=data_df, save_model_to="./saved_models/MLP_model_full_data.pth"
            )
        elif MODE == Mode.TRAIN_ON_SPLIT_DATA:
            model = train_MLP_model(
                train_df,
                test_df,
                save_model_to="./saved_models/MLP_model_split_data.pth",
            )
        elif MODE == Mode.LOAD_MODEL_FOR_FULL_DATA:
            model = load_MLP_model(
                "./saved_models/MLP_model_full_data.pth", data_df, load_full_model=True
            )
        else:
            model = load_MLP_model(
                "./saved_models/MLP_model_split_data.pth",
                test_df,
                load_full_model=False,
            )
    elif MODEL == ModelType.LINEAR_REGRESSION:
        if MODE == Mode.TRAIN_ON_FULL_DATA:
            model = train_linear_regression_model(
                data_df,
                save_model_to="./saved_models/linear_regression_model_full_data.pkl",
            )
        elif MODE == Mode.TRAIN_ON_SPLIT_DATA:
            model = train_linear_regression_model(
                train_df,
                test_df,
                save_model_to="./saved_models/linear_regression_model_split_data.pkl",
            )
        elif MODE == Mode.LOAD_MODEL_FOR_FULL_DATA:
            model = load_linear_regression_model(
                "./saved_models/linear_regression_model_full_data.pkl"
            )
        else:
            model = load_linear_regression_model(
                "./saved_models/linear_regression_model_split_data.pkl", test_df
            )
    else:
        model = train_item_based_recommendation_model(
            train_df,
            user_profiles,
            all_item_profiles,
            user_encoder,
            item_encoder,
            calc_metrics=(
                MODE == Mode.TRAIN_ON_SPLIT_DATA
                or MODE == Mode.LOAD_MODEL_FOR_SPLIT_DATA
            ),
        )

    user_encoded_label = user_encoder.transform([USER_OBJECTID])[0]
    positive_user_interactions = test_df.loc[
        (test_df["user_objectid"] == user_encoded_label) & (test_df["label"] != 0)
    ]
    assert (
        user_encoded_label in positive_user_interactions.loc[:, "user_objectid"].values
    ), (
        f"'user_objectid' {positive_user_interactions} not found in positive interactions of the test set."
    )

    X_test_unseen_items = get_subset_by_column_substring(
        positive_user_interactions, "item_"
    )
    y_test_unseen_items = positive_user_interactions["label"]
    live_unseen_items = live_item_profiles.copy()
    live_unseen_items["item_objectid"] = item_encoder.transform(
        live_unseen_items["item_objectid"]
    )

    previous_interactions = train_df.loc[
        train_df["user_objectid"] == user_encoded_label
    ]
    item_objectids = item_encoder.inverse_transform(
        previous_interactions.loc[:, "item_objectid"].values
    )
    previous_interactions = previous_interactions.drop(columns=["item_objectid"])
    previous_interactions["item_objectid"] = item_objectids

    if MODE == Mode.TRAIN_ON_FULL_DATA or MODE == Mode.LOAD_MODEL_FOR_FULL_DATA:
        evaluate_user_recommendations(
            USER_OBJECTID,
            live_unseen_items,
            model,
            user_encoder,
            item_encoder,
            user_profiles,
            all_item_profiles,
            previous_interactions=previous_interactions,
            top_n=TOP_N_RECOMMENDATIONS,
        )
    else:
        evaluate_user_recommendations(
            USER_OBJECTID,
            X_test_unseen_items,
            model,
            user_encoder,
            item_encoder,
            user_profiles,
            all_item_profiles,
            previous_interactions=previous_interactions,
            ground_truth_for_unseen_items=y_test_unseen_items,
            top_n=TOP_N_RECOMMENDATIONS,
        )


if __name__ == "__main__":
    main()

