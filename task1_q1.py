import pandas as pd


def find_user_with_most_interactions(
    csv_path: str, interaction_type: str
) -> tuple[str | None, int]:
    """
    Finds the user with the most interactions of a specific type from CSV data.

    Args:
        csv_path (str): Path to CSV data.
        interaction_type (str): The type of interaction to count (e.g., 'wishlist', 'order').

    Returns:
        tuple: A tuple containing the user ID with the most interactions of the specified type
               and the count of those interactions.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV data: {e}")
        return None, -1

    # Filter the DataFrame for the specified interaction type
    filtered_interactions = df[df["interaction_type"] == interaction_type]

    if filtered_interactions.empty:
        return None, 0

    # Count the number of interactions for each user
    interaction_counts = filtered_interactions.loc[:, "user_objectid"].value_counts()

    # Find the user with the most interactions
    most_interacting_user = interaction_counts.idxmax()
    most_interaction_count = interaction_counts.max()

    return most_interacting_user, most_interaction_count


def main() -> None:
    path = "data/user_item_interactions.csv"
    most_wishlist_user, most_wishlist_count = find_user_with_most_interactions(
        path, "wishlist"
    )

    print(f"User with the most wishlist interactions: {most_wishlist_user}")
    print(f"Number of wishlist interactions: {most_wishlist_count}")


if __name__ == "__main__":
    main()
