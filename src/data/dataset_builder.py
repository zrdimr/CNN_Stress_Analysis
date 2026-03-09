import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN 

class StressDataset(Dataset):
    def __init__(self, X, y, model_type="cnn1d"):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.model_type = model_type

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_val = self.X[idx]
        
        # Format input based on CNN requirements
        if self.model_type == "cnn2d":
            # For 34 features, we pad with 2 zeros to get 36, then reshape to 1 x 6 x 6
            if x_val.shape[0] < 36:
                pad = torch.zeros(36 - x_val.shape[0])
                x_val = torch.cat([x_val, pad])
            x_val = x_val[:36].view(1, 6, 6)
        elif self.model_type in ["cnn1d", "fusion"]:
            # (1, sequence_length)
            x_val = x_val.unsqueeze(0)
            
        return x_val, self.y[idx]

def build_dataloaders(config, model_type, balancing_strategy='none'):
    train_df = pd.read_csv(config['data']['train_path']).dropna()
    test_df = pd.read_csv(config['data']['test_path']).dropna()

    target_col = config['data']['target_col']
    drop_cols = config['data']['drop_cols'] + [target_col]

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[target_col]

    # Normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Dataset Balancing (Data Augmentation)
    if balancing_strategy == "smote":
        print(f"Applying SMOTE...")
        sampler = SMOTE(random_state=42)
        X_train_scaled, y_train_enc = sampler.fit_resample(X_train_scaled, y_train_enc)
    elif balancing_strategy == "adasyn":
        print(f"Applying ADASYN...")
        sampler = ADASYN(random_state=42)
        X_train_scaled, y_train_enc = sampler.fit_resample(X_train_scaled, y_train_enc)
    else:
        print(f"No balancing applied...")

    train_dataset = StressDataset(X_train_scaled, y_train_enc, model_type)
    test_dataset = StressDataset(X_test_scaled, y_test_enc, model_type)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader, le.classes_, X_train_scaled.shape[1]
