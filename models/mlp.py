import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.preprocess import get_subset_by_column_substring

class RecommendationDataset(Dataset):
    def __init__(self, user_features: np.ndarray, item_features: np.ndarray, labels: np.ndarray) -> None:
        self.user_features = user_features
        self.item_features = item_features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_feature = self.user_features[idx]
        item_feature = self.item_features[idx]
        label = self.labels[idx]
        return torch.tensor(user_feature, dtype=torch.float), torch.tensor(item_feature, dtype=torch.float), torch.tensor(label, dtype=torch.float)


class MLP(torch.nn.Module):
    def __init__(self, num_user_features: int, num_item_features: int, hidden_dims: list[int] = [128, 64]) -> None:
        super(MLP, self).__init__()
        self.user_fc = torch.nn.Linear(num_user_features, hidden_dims[0])
        self.item_fc = torch.nn.Linear(num_item_features, hidden_dims[0])
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(hidden_dims[0] * 2, hidden_dims[1])
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(0.2)
        self.fc3 = torch.nn.Linear(hidden_dims[1], 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        user_embedded = self.user_fc(user_features)
        item_embedded = self.item_fc(item_features)
        x = torch.cat((user_embedded, item_embedded), dim=1)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x.squeeze()
    

def train_MLP_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame = None,
    save_model_to: str = None
) -> MLP:
    # Hyperparameters
    learning_rate = 0.001
    epochs = 10
    batch_size = 100
    
    train_user_features_df = get_subset_by_column_substring(train_df, "user_")
    train_item_features_df = get_subset_by_column_substring(train_df, "item_")
    y_train = train_df["label"]
    num_user_features = train_user_features_df.shape[1]
    num_item_features = train_item_features_df.shape[1]
    
    
    
    train_dataset = RecommendationDataset(
        train_user_features_df.values,
        train_item_features_df.values,
        y_train
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    
    model = MLP(num_user_features, num_item_features)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for user_features, item_features, labels in train_loader:
            optimizer.zero_grad()
            predictions = model(user_features, item_features)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    model.eval()
    if test_df is not None:
        test_user_features_df = get_subset_by_column_substring(test_df, "user_")
        test_item_features_df = get_subset_by_column_substring(test_df, "item_")
        y_test = test_df["label"]
        with torch.no_grad():
            y_pred = model(
                torch.tensor(test_user_features_df.values, dtype=torch.float),
                torch.tensor(test_item_features_df.values, dtype=torch.float)
            )
            y_true = torch.tensor(y_test.values, dtype=torch.float)
            
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            print("Model Performance Metrics:")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"Mean Squared Error (MSE): {mse:.4f}")
    
    if save_model_to is not None:
        torch.save(model.state_dict(), save_model_to)
        print(f"Model saved as '{save_model_to}'")
    
    return model


def load_MLP_model(filename: str, data: pd.DataFrame, load_full_model: bool = True) -> MLP:
    user_features_df = get_subset_by_column_substring(data, "user_")
    item_features_df = get_subset_by_column_substring(data, "item_")
    num_user_features = user_features_df.shape[1]
    num_item_features = item_features_df.shape[1]
    
    model = MLP(num_user_features, num_item_features)
    model.load_state_dict(torch.load(filename))
    print(f"Model loaded from '{filename}'")
    model.eval()
    
    if not load_full_model:
        y_test = data["label"]
        with torch.no_grad():
            y_pred = model(
            torch.tensor(user_features_df.values, dtype=torch.float),
            torch.tensor(item_features_df.values, dtype=torch.float)
            )
            y_true = torch.tensor(y_test.values, dtype=torch.float)

            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            print("Model Performance Metrics:")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"Mean Squared Error (MSE): {mse:.4f}")
    
    return model