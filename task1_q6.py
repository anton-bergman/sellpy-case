import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_ordered_and_wishlisted_items_per_country(
    user_item_interactions_csv_path: str, user_features_csv_path: str
) -> dict[str, dict[str, int]]:
    """
    Counts the number of ordered and wishlisted items per country.

    Args:
        user_item_interactions_csv_path (str): Path to the CSV file containing user interactions.
        user_features_csv_path (str): Path to the CSV file containing user features.

    Returns:
        dict: A dictionary where keys are country codes and values are dictionaries with counts of ordered and wishlisted items.
    """
    try:
        user_item_interactions_df = pd.read_csv(user_item_interactions_csv_path)
        user_features_df = pd.read_csv(user_features_csv_path)
    except Exception as e:
        print(f"Error reading CSV data: {e}")
        return {}

    # Inner join on user_objectid to get both interaction_type and country
    merged_df = pd.merge(
        user_item_interactions_df, user_features_df, on="user_objectid", how="inner"
    )

    # Group by country and interaction type
    grouped = (
        merged_df.groupby(["country", "interaction_type"]).size().unstack(fill_value=0)
    )

    return grouped.to_dict()


def main() -> None:
    user_item_interactions_path = "data/user_item_interactions.csv"
    user_features_path = "data/user_features.csv"

    country_interactions = get_ordered_and_wishlisted_items_per_country(
        user_item_interactions_path, user_features_path
    )
    print(f"Ordered and wishlisted items per country: {country_interactions}")

    orders_per_country = country_interactions["order"]
    wishlists_per_country = country_interactions["wishlist"]
    countries = list(orders_per_country.keys())
    orders = [orders_per_country.get(country, 0) for country in countries]
    wishlists = [wishlists_per_country.get(country, 0) for country in countries]
    print(f"Countries: {countries}")
    print(f"Orders: {orders}")
    print(f"Wishlists: {wishlists}")

    width = 0.4
    half_width = width / 2
    x = np.arange(len(countries))
    y1 = orders
    y2 = wishlists
    plt.bar(x - half_width, y1, width, label="Orders", color="blue")
    plt.bar(x + half_width, y2, width, label="Wishlists", color="orange")
    plt.xticks(x, countries)
    plt.xlabel("Countries")
    plt.ylabel("Number of Interactions")
    plt.legend(["Orders", "Wishlists"])
    plt.title("Ordered and Wishlisted Items per Country")
    plt.show()

    # Note: Unclear if the graph should be a bar chart with two bars per country displaying orders
    # and wishlists, or if it should be a bar chart with one bar per country showing the number of
    # interactions with items that are both ordered and wishlisted.


if __name__ == "__main__":
    main()
