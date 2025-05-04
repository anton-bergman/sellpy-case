import ast

import pandas as pd


def get_num_live_items_by_type(live_item_data_csv_path: str, target_type: str) -> int:
    """
    Counts the number of live items of a specific type in the user features CSV.

    Args:
        user_features_csv_path (str): Path to the CSV file containing user features.
        target_type (str): The type of item to count (e.g., 'Jeans').

    Returns:
        int: The number of live items of the specified type.
    """
    try:
        user_features_df = pd.read_csv(live_item_data_csv_path)
    except Exception as e:
        print(f"Error reading CSV data: {e}")
        return -1

    num_live_items_of_target_type = 0
    for _, row in user_features_df.iterrows():
        metadata_str = row.loc["metadata"]
        metadata = ast.literal_eval(metadata_str)

        if metadata["type"] == target_type:
            num_live_items_of_target_type += 1

    return num_live_items_of_target_type


def main() -> None:
    live_item_data_csv_path = "data/live_item_data.csv"
    num_live_items_of_target_type = get_num_live_items_by_type(
        live_item_data_csv_path, "Jeans"
    )
    print(f"Number of live items of type 'Jeans': {num_live_items_of_target_type}")


if __name__ == "__main__":
    main()
