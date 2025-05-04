import pandas as pd

def get_top_n_item_categories(user_item_interactions_csv_path, item_data_for_interactions_csv_path, top_n=3) -> list[tuple[str, int]]:
    """
    Calculates the top N most popular item categories from user interactions.

    Args:
        user_interactions_csv_path (str): Path to the CSV file containing user interactions.
        item_categories_csv_path (str): Path to the CSV file containing item categories.
        top_n (int): The number of top categories to return.

    Returns:
        list: A list of tuples, where each tuple contains a category and its interaction count,
              sorted in descending order of count.  Returns an empty list on error.
    """
    try:
        user_item_interactions_df = pd.read_csv(user_item_interactions_csv_path)
        item_data_df = pd.read_csv(item_data_for_interactions_csv_path)
    except Exception as e:
        print(f"Error reading CSV data: {e}")
        return []

    # Inner join to get item categories
    merged_df = pd.merge(user_item_interactions_df, item_data_df, on="item_objectid", how="inner")

    # Get top 3 categories
    category_counts = merged_df["category"].value_counts()
    top_3_categories = category_counts.head(top_n).items()

    return list(top_3_categories)


def main() -> None:
    user_item_interactions_path = "data/user_item_interactions.csv"
    item_data_for_interactions_path = "data/item_data_for_interactions.csv"
    
    category_interactions = get_top_n_item_categories(user_item_interactions_path, item_data_for_interactions_path, 3)
    print(f"Category interactions: {category_interactions}")
    

if __name__ == "__main__":
    main()
