import ast
import pandas as pd

def analyze_demographic_interest(user_features_csv_path: str, target_demographic: str) -> tuple[int, int]:
    """
    Analyzes user features data to determine the share of users interested in a specific demographic
    and the number of users for whom it is the primary interest.

    Args:
        user_features_csv_path: Path to the user_features.csv file.
        target_demographic: The demographic to analyze (e.g., 'Kids', 'Women').

    Returns:
        A tuple containing:
            - The number of users interested in the target demographic.
            - The number of users for whom the target demographic is the primary interest.
    """
    try:
        user_features_df = pd.read_csv(user_features_csv_path)
    except Exception as e:
        print(f"Error reading CSV data: {e}")
        return -1
    
    num_users_with_interest = 0
    num_users_with_primary_interest = 0
    for _, row in user_features_df.iterrows():
        demography_array_str = row["demography_array"]
        demography_array = ast.literal_eval(demography_array_str)
        
        if target_demographic in demography_array:
            num_users_with_interest += 1
            if demography_array[0] == target_demographic:
                # If the target demographic is the first element, it is considered the primary interest
                num_users_with_primary_interest += 1
    
    return num_users_with_interest, num_users_with_primary_interest


def main() -> None:
    user_features_path = "data/user_features.csv"
    num_users_with_interest, num_users_with_primary_interest = analyze_demographic_interest(user_features_path, "Kids")
    
    print(f"Number of users with 'Kids' interest: {num_users_with_interest}")
    print(f"Number of users with 'Kids' as main interest: {num_users_with_primary_interest}")
    

if __name__ == "__main__":
    main()
