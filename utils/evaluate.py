import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import torch
from models.mlp import MLP
from sklearn.preprocessing import LabelEncoder
from utils.preprocess import parse_demographic_or_brand, get_subset_by_column_substring

def evaluate_recommendation(
    user_objectid: str,
    item_objectid: str,
    user_profiles: pd.DataFrame,
    all_item_profiles: pd.DataFrame,
    previous_interactions: pd.DataFrame = None,
) -> None:
    """Evaluates the recommendation for a specific user and item.
    
    Args:
        user_objectid (str): The user object ID.
        item_objectid (str): The item object ID.
        user_profiles (pd.DataFrame): DataFrame containing user profiles.
        all_item_profiles (pd.DataFrame): DataFrame containing all item profiles.
        previous_interactions (pd.DataFrame, optional): DataFrame containing previous interactions. Defaults to None.
    """
    item_profile = all_item_profiles[all_item_profiles["item_objectid"] == item_objectid]
    assert item_profile.shape[0] == 1, f"Item {item_objectid} did not return a single row from all_item_profiles."
    item_profile_no_id_column = item_profile.drop(columns=["item_objectid"])
    item_profile_no_id_column = item_profile_no_id_column.iloc[0]
    
    item_features = {}
    for col_name, value in item_profile_no_id_column.items():
        if value != 0:
            item_features[col_name] = float(value)
    assert len(item_features) == 2, f"Item {item_objectid} shoud have exactly one brand and one demography, not {item_features}."
    item_feature_names = list(item_features.keys())
    is_demography_first = "demography" in item_feature_names[0]
    item_demography = item_feature_names[0] if is_demography_first else item_feature_names[1]
    item_demography = parse_demographic_or_brand(item_demography)
    item_brand = item_feature_names[1] if is_demography_first else item_feature_names[0]
    item_brand = parse_demographic_or_brand(item_brand)
    
    user_features = {}
    user_profile = user_profiles[user_profiles["user_objectid"] == user_objectid]
    assert user_profile.shape[0] == 1, f"User {user_objectid} did not return a single row from user_profiles."
    user_profile_no_id_column = user_profile.drop(columns=["user_objectid"])
    user_profile_no_id_column = user_profile.iloc[0]
        
    
    all_user_features = user_profile_no_id_column.to_dict()
    user_features[f"user_demography_['{item_demography}']"] = float(all_user_features[f"user_demography_['{item_demography}']"])
    user_features[f"user_brand_['{item_brand}']"] = float(all_user_features[f"user_brand_['{item_brand}']"])
    print(f"User {user_objectid} has the following relevant features for item {item_objectid}: {user_features}")
    
    if previous_interactions is not None:
        num_previous_interactions_with_item_demography = 0
        num_previous_interactions_with_item_brand = 0
        for _, row in previous_interactions.iterrows():
            prev_item_df: pd.DataFrame = all_item_profiles[all_item_profiles["item_objectid"] == row["item_objectid"]]
            assert prev_item_df.shape[0] == 1, f"Item {row['item_objectid']} did not return a single row from all_item_profiles."
            prev_item_profile_no_id_column = prev_item_df.drop(columns=["item_objectid"])
            prev_item_profile_no_id_column = prev_item_profile_no_id_column.iloc[0]
            
            prev_item_features = {}
            for col_name, value in prev_item_profile_no_id_column.items():
                if value != 0:
                    prev_item_features[col_name] = value
            assert len(prev_item_features) == 2, f"Item {row['item_objectid']} shoud have exactly one brand and one demography, not {prev_item_features}."
            prev_item_feature_names = list(prev_item_features.keys())
            is_demography_first = "demography" in prev_item_feature_names[0]
            prev_item_demography = prev_item_feature_names[0] if is_demography_first else prev_item_feature_names[1]
            prev_item_brand = prev_item_feature_names[1] if is_demography_first else prev_item_feature_names[0]
            
            if parse_demographic_or_brand(prev_item_demography) == item_demography:
                num_previous_interactions_with_item_demography += 1
            if parse_demographic_or_brand(prev_item_brand) == item_brand:
                num_previous_interactions_with_item_brand += 1
        
        print(f"User {user_objectid} has {num_previous_interactions_with_item_demography} previous interactions with: {item_demography}.")
        print(f"User {user_objectid} has {num_previous_interactions_with_item_brand} previous interactions with: {item_brand}.")
    return

def evaluate_user_recommendations(
    user_objectid: str,
    unseen_items: pd.DataFrame,
    model: np.ndarray | LinearRegression | MLP,
    user_encoder: LabelEncoder,
    item_encoder: LabelEncoder,
    user_profiles: pd.DataFrame,
    all_item_profiles: pd.DataFrame,
    previous_interactions: pd.DataFrame = None,
    ground_truth_for_unseen_items: pd.DataFrame = None,
    top_n: int = 10
) -> None:
    """Evaluates recommendations for a specific user and unseen items.
    
    Args:
        user_objectid (str): The user object ID.
        unseen_items (pd.DataFrame): DataFrame containing unseen items for the user.
        model (LinearRegression): Trained linear regression model.
        user_encoder (LabelEncoder): Encoder for user object IDs.
        item_encoder (LabelEncoder): Encoder for item object IDs.
        user_profiles (pd.DataFrame): DataFrame containing user profiles.
        all_item_profiles (pd.DataFrame): DataFrame containing all item profiles.
        previous_interactions (pd.DataFrame, optional): DataFrame containing previous interactions. Defaults to None.
        ground_truth_for_unseen_items (pd.DataFrame, optional): Ground truth labels for unseen items. Defaults to None.
        top_n (int, optional): Number of top recommendations to display. Defaults to 10.
    """
    user_columns = [col_name for col_name in unseen_items.columns if "user" in col_name]
    assert not user_columns, f"Unseen items DataFrame should not contain any user features, only item features. User features in unseen_items: {user_columns}"
    user_profile = user_profiles[user_profiles["user_objectid"] == user_objectid].copy()
    assert user_profile.shape[0] == 1, f"User {user_objectid} did not return a single row from user_profiles."
    user_encoded_label = user_encoder.transform([user_objectid])[0]
    user_profile.iloc[0, user_profile.columns.get_loc("user_objectid")] = user_encoded_label

    user_profile_np = user_profile.iloc[0].to_numpy()
    repeated_user_profile = np.repeat(user_profile_np.reshape(1, -1), unseen_items.shape[0], axis=0)
    user_profile = pd.DataFrame(repeated_user_profile, columns=user_profile.columns, dtype=float)
    user_profile["user_objectid"] = user_profile["user_objectid"].astype(np.int64)
    
    unseen_items = unseen_items.reset_index(drop=True)
    unseen_data = pd.concat([user_profile, unseen_items], axis=1)
    unseen_data = unseen_data.reindex(columns=sorted(list(unseen_data.columns)))
    
    assert unseen_data["user_objectid"].dtype.type is np.int64, f"'user_objectid' should be of type 'np.int64', not {unseen_data["user_objectid"].dtype.type}."
    assert unseen_data["item_objectid"].dtype.type is np.int64, f"'item_objectid' should be of type 'np.int64', not {unseen_data["item_objectid"].dtype.type}."
    unseen_items_objectids = item_encoder.inverse_transform(unseen_items["item_objectid"].values).tolist()
    
    # Predict recommendation scores for unseen items
    assert "label" not in unseen_data.columns, "Unseen data DataFrame should not contain a 'label' column."
    if isinstance(model, LinearRegression):
        pred = model.predict(unseen_data).tolist()
    elif isinstance(model, np.ndarray):
        item_encoded_label = unseen_data["item_objectid"].values
        pred = model[user_encoded_label, item_encoded_label].tolist()
    elif isinstance(model, MLP):
        unseen_user_features_df = get_subset_by_column_substring(unseen_data, "user_")
        unseen_item_features_df = get_subset_by_column_substring(unseen_data, "item_")
        unseen_user_features = torch.tensor(unseen_user_features_df.values, dtype=torch.float)
        unseen_item_features = torch.tensor(unseen_item_features_df.values, dtype=torch.float)
        with torch.no_grad():
            pred = model(unseen_user_features, unseen_item_features).numpy().tolist()
    else:
        raise AssertionError(f"'model' must be of some type: {[np.ndarray, LinearRegression, MLP]}. Not {type(model)}")
    
    # Combine results and sort the predicted recommendation scores
    if ground_truth_for_unseen_items is not None:
        result = list(zip(unseen_items_objectids, ground_truth_for_unseen_items.values.tolist(), pred))
    else:
        result = list(zip(unseen_items_objectids, pred))
    ranked_result = sorted(result, key=lambda x: x[-1], reverse=True)

    # Print the top N recommendations
    print()
    for n in range(min(top_n, len(ranked_result))):
        is_last_iteration = n == min(top_n, len(ranked_result))-1
        nth_recommendation = ranked_result[n]
        
        if ground_truth_for_unseen_items is not None:
            nth_recommendation_dict = dict(zip(["item_objectid", "label", "score"], nth_recommendation))
        else:
            nth_recommendation_dict = dict(zip(["item_objectid", "score"], nth_recommendation))
        print(f"Recommendation {n+1} for user {user_objectid}: {nth_recommendation_dict}")
        
        if previous_interactions is not None:
            evaluate_recommendation(
                user_objectid,
                nth_recommendation[0],
                user_profiles,
                all_item_profiles,
                previous_interactions
            )
            if not is_last_iteration:
                print()