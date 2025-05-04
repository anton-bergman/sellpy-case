import ast
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def read_csv_data(file_path: str) -> pd.DataFrame:
    """Reads a CSV file and returns a DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the data from the CSV file.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV data: {e}")
        return pd.DataFrame()
    return df


def build_user_profile(user_features_df: pd.DataFrame) -> pd.DataFrame:
    """Builds user profiles from the user features dataframe.

    Args:
        user_features_df (pd.DataFrame): DataFrame containing user features.

    Returns:
        pd.DataFrame: A DataFrame where each row represents a user profile with scores for different features.
    """

    def build_row_user_profile(row: pd.Series) -> dict[str, str]:
        profile = {}

        # Demographies
        row_demography_array = ast.literal_eval(row.loc["demography_array"])
        row_score_array_demography = ast.literal_eval(row.loc["score_array_demography"])
        for demography, score in zip(row_demography_array, row_score_array_demography):
            profile[f"user_demography_['{demography}']"] = score

        # Brands
        row_brand_array = ast.literal_eval(row.loc["brand_array"])
        row_score_array_brand = ast.literal_eval(row.loc["score_array_brand"])
        for brand, score in zip(row_brand_array, row_score_array_brand):
            profile[f"user_brand_['{brand}']"] = score
        return profile

    user_profiles = user_features_df.apply(build_row_user_profile, axis=1)
    user_profiles = pd.DataFrame(user_profiles.tolist()).fillna(0)
    user_profiles.insert(0, "user_objectid", user_features_df["user_objectid"].values)
    return user_profiles


def build_item_profile(item_data_df: pd.DataFrame) -> pd.DataFrame:
    """Builds item profiles from the live item data dataframe.
    Args:
        item_data_df (pd.DataFrame): DataFrame containing live item data.

    Returns:
        pd.DataFrame: A DataFrame where each row represents an item profile with scores for different features.
    """

    def build_row_item_profile(row: pd.Series) -> dict[str, str]:
        profile = {}
        # If metadata is NaN, set it to an empty dictionary
        row_metadata = (
            {}
            if pd.isna(row.loc["metadata"])
            else ast.literal_eval(row.loc["metadata"])
        )

        if "demography" in row_metadata.keys():
            profile[f"item_demography_['{row_metadata['demography']}']"] = 1
        else:
            profile["item_demography_['No demography']"] = 1
        if "brand" in row_metadata.keys():
            profile[f"item_brand_['{row_metadata['brand']}']"] = 1
        else:
            profile["item_brand_['No brand']"] = 1
        return profile

    item_profiles = item_data_df.apply(build_row_item_profile, axis=1)
    item_profiles = pd.DataFrame(item_profiles.tolist()).fillna(0)
    item_profiles.insert(0, "item_objectid", item_data_df["item_objectid"].values)
    return item_profiles


def get_all_demographies_and_brands(
    user_features_df: pd.DataFrame,
    item_data_for_interactions_df: pd.DataFrame,
    live_item_data_df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    demographies = set()
    brands = set()
    for _, row in user_features_df.iterrows():
        demography_array = ast.literal_eval(row.loc["demography_array"])
        demographies.update(demography_array)
        brand_array = ast.literal_eval(row.loc["brand_array"])
        brands.update(brand_array)

    for _, row in item_data_for_interactions_df.iterrows():
        row_metadata = (
            {}
            if pd.isna(row.loc["metadata"])
            else ast.literal_eval(row.loc["metadata"])
        )
        if "demography" in row_metadata.keys():
            demographies.add(row_metadata["demography"])
        else:
            demographies.add("No demography")
        if "brand" in row_metadata.keys():
            brands.add(row_metadata["brand"])
        else:
            brands.add("No brand")

    for _, row in live_item_data_df.iterrows():
        row_metadata = (
            {}
            if pd.isna(row.loc["metadata"])
            else ast.literal_eval(row.loc["metadata"])
        )
        if "demography" in row_metadata.keys():
            demographies.add(row_metadata["demography"])
        else:
            demographies.add("No demography")
        if "brand" in row_metadata.keys():
            brands.add(row_metadata["brand"])
        else:
            brands.add("No brand")

    demographies = sorted(list(demographies))
    brands = sorted(list(brands))
    return demographies, brands


def parse_demographic_or_brand(input_string: str):
    """
    Parses a string to extract the demographic or brand information.

    Args:
        input_string (str): A string formatted as "item_demography_['Women']",
                        "user_demography_['Women']", "user_brand_['Nike']", or
                        "item_brand_['Nike']".

    Returns:
        str: The extracted demographic or brand (e.g., 'Women', 'Nike').
               Returns None if the string is not in the expected format.
    """
    parts = input_string.split("[")
    if len(parts) != 2:
        raise AssertionError(
            f"Input string: '{input_string}', does not follow the expected format."
        )

    demographic_or_brand = parts[1][1:-2]
    return demographic_or_brand


def get_preprocessed_data(
    user_features_df: pd.DataFrame,
    item_data_for_interactions_df: pd.DataFrame,
    live_item_data_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Preprocess raw data."""
    # Build profiles
    user_profiles = build_user_profile(user_features_df)
    item_profiles = build_item_profile(item_data_for_interactions_df)
    live_item_profiles = build_item_profile(live_item_data_df)

    # Combine all item profiles in one DataFrame
    all_item_features = sorted(
        list(set(item_profiles.columns) | set(live_item_profiles.columns))
    )
    item_profiles_extended = item_profiles.reindex(
        columns=all_item_features, fill_value=float(0)
    )
    live_item_profiles_extended = live_item_profiles.reindex(
        columns=all_item_features, fill_value=float(0)
    )
    all_item_profiles = pd.concat(
        [item_profiles_extended, live_item_profiles_extended], axis=0
    ).reset_index(drop=True)
    all_item_profiles = all_item_profiles.drop_duplicates(keep="first")

    all_demographies, all_brands = get_all_demographies_and_brands(
        user_features_df, item_data_for_interactions_df, live_item_data_df
    )
    all_user_features = sorted(
        ["user_demography_['" + s + "']" for s in all_demographies]
        + ["user_brand_['" + s + "']" for s in all_brands]
        + ["user_objectid"]
    )
    all_item_features = sorted(
        ["item_demography_['" + s + "']" for s in all_demographies]
        + ["item_brand_['" + s + "']" for s in all_brands]
        + ["item_objectid"]
    )
    user_profiles = user_profiles.reindex(columns=all_user_features, fill_value=0)
    item_profiles = item_profiles.reindex(columns=all_item_features, fill_value=0)
    live_item_profiles = live_item_profiles.reindex(
        columns=all_item_features, fill_value=0
    )
    all_item_profiles = all_item_profiles.reindex(
        columns=all_item_features, fill_value=0
    )
    return user_profiles, item_profiles, live_item_profiles, all_item_profiles


def get_dataset(
    user_features_df: pd.DataFrame,
    user_item_interactions_df: pd.DataFrame,
    item_data_for_interactions_df: pd.DataFrame,
    live_item_data_df: pd.DataFrame,
    user_profiles: pd.DataFrame,
    item_profiles: pd.DataFrame,
    live_item_profiles: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """Create a traning dataset by getting positive and generating negative samples."""
    positive_df = get_positive_samples(user_item_interactions_df)
    negative_df = generate_negative_samples(
        user_item_interactions_df, num_negatives_per_positive=1
    )

    merged_df = pd.concat([positive_df, negative_df], ignore_index=True)
    merged_df = merged_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    data_df = pd.merge(merged_df, user_profiles, on="user_objectid")
    data_df = pd.merge(data_df, item_profiles, on="item_objectid")

    # Align feature space for dataset
    all_features = sorted(
        list(
            set(user_profiles.columns)
            | set(item_profiles.columns)
            | set(live_item_profiles.columns)
            | {"label"}
        )
    )
    data_df = data_df.reindex(columns=all_features, fill_value=float(0))

    # Encode all user and item IDs from string-IDs to int-IDs
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    all_users = user_features_df["user_objectid"]
    all_items = sorted(
        list(
            set(item_data_for_interactions_df["item_objectid"].values)
            | set(live_item_data_df["item_objectid"].values)
        )
    )
    user_encoder.fit(all_users)
    item_encoder.fit(all_items)
    data_df["user_objectid"] = user_encoder.transform(data_df["user_objectid"])
    data_df["item_objectid"] = item_encoder.transform(data_df["item_objectid"])

    return data_df, user_encoder, item_encoder


def generate_negative_samples(
    interactions_df: pd.DataFrame, num_negatives_per_positive: int = 1
) -> pd.DataFrame:
    """Generates negative samples by pairing users with non-interacted items."""
    unique_users = sorted(list(interactions_df["user_objectid"].unique()))
    unique_items = sorted(list(interactions_df["item_objectid"].unique()))

    negative_samples = []

    for user in unique_users:
        user_positive_items = sorted(
            list(
                interactions_df.loc[interactions_df["user_objectid"] == user][
                    "item_objectid"
                ].unique()
            )
        )
        possible_negatives_items = sorted(
            list(set(unique_items) - set(user_positive_items))
        )

        if num_negatives_per_positive <= len(possible_negatives_items):
            negative_items = random.sample(
                possible_negatives_items,
                num_negatives_per_positive * len(user_positive_items),
            )
            for item in negative_items:
                negative_samples.append(
                    {"user_objectid": user, "item_objectid": item, "label": 0.0}
                )

    negative_df = pd.DataFrame(negative_samples)
    return negative_df


def get_positive_samples(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """Creates a DataFrame with positive samples based on interaction type.

    Args:
        interactions_df (pd.DataFrame): DataFrame containing user-item interactions with an "interaction_type" column.

    Returns:
        pd.DataFrame: DataFrame with columns "user_objectid", "item_objectid", and "label".
    """
    # Filter for rows where interaction_type is "order" or "wishlist"
    positive_samples = interactions_df.copy()

    # Assign labels based on interaction_type
    # positive_samples["label"] = positive_samples["interaction_type"].map(
    #     {"order": 1.0, "wishlist": 0.7}
    # )
    positive_samples["label"] = positive_samples["interaction_type"].map(
        lambda x: 1.0 if x == "order" else 0.7 if x == "wishlist" else np.nan
    )

    # Select only the required columns
    positive_samples = positive_samples.loc[
        :, ["user_objectid", "item_objectid", "label"]
    ]
    assert positive_samples.notna().all().all(), "There are NaN values in the DataFrame"

    return positive_samples


def get_subset_by_column_substring(
    df: pd.DataFrame, column_name_substring: str
) -> pd.DataFrame:
    """
    Fetches a subset of the DataFrame containing only the columns
    whose names contain a given substring.

    Args:
        df (pd.DataFrame): The input pandas DataFrame.
        column_name_substring (str): The substring to search for within the column names.

    Returns:
        pd.DataFrame: A new DataFrame containing only the columns that have the substring in their
        names. Returns an empty DataFrame if no columns match.
    """
    matching_columns = [col for col in df.columns if column_name_substring in col]
    return df.loc[:, matching_columns]

