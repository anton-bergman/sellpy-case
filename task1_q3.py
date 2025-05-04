import pandas as pd

def get_num_users_per_country(user_features_csv_path: str, country_code) -> int:
    """
    Counts the number of users from a specific country in the user features CSV.

    Args:
        user_features_csv_path (str): Path to the CSV file containing user features.
        country_code (str): The country code to filter users by.

    Returns:
        int: The number of users from the specified country.
    """
    try:
        user_features_df = pd.read_csv(user_features_csv_path)
    except Exception as e:
        print(f"Error reading CSV data: {e}")
        return -1
    
    return user_features_df[user_features_df["country"] == country_code].shape[0]


def main() -> None:
    user_features_path = "data/user_features.csv"
    num_german_users = get_num_users_per_country(user_features_path, "DE")
    print(f"Number of users from Germany: {num_german_users}")
    

if __name__ == "__main__":
    main()
