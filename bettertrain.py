# train_resnet50_meta_boosted.py
import os
import h5py
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import autocast
import torch.nn.utils as nn_utils
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ========== CONFIG ==========
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
TRAIN_CSV = "/Users/santiagovillasenor/Library/CloudStorage/Dropbox-HeuristicsFinansoft/Jaime Villasenor/Personal/Santi/ITAM/8_Semestre/Neural Networks/Kaggle/KaggleSubmision/data/new-train-metadata.csv"
TRAIN_HDF5 = "/Users/santiagovillasenor/Library/CloudStorage/Dropbox-HeuristicsFinansoft/Jaime Villasenor/Personal/Santi/ITAM/8_Semestre/Neural Networks/Kaggle/KaggleSubmision/data/train-image.hdf5"

POS_RATIO = 0.26
NUM_FOLDS = 10
TOP_K = 5
SEED = 42
PATIENCE = 7

# ========== DATASET ==========
class ISIC_HDF5_MetaDataset(Dataset):
    def __init__(self, df, hdf5_path, transform=None):
        self.df = df.reset_index(drop=True)
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.meta = self._preprocess_metadata(df)

    def _preprocess_metadata(self, df):
        df = df.copy()
        df['sex'] = LabelEncoder().fit_transform(df['sex'].fillna("unknown"))
        df['anatom_site_general'] = LabelEncoder().fit_transform(df['anatom_site_general'].fillna("unknown"))
        df['age_approx'] = df['age_approx'].fillna(df['age_approx'].median())
        meta_features = df[['age_approx', 'sex', 'anatom_site_general']].values
        return torch.tensor(StandardScaler().fit_transform(meta_features), dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        isic_id = row["isic_id"]
        label = torch.tensor(row["target"], dtype=torch.float32)
        meta = self.meta[idx]

        with h5py.File(self.hdf5_path, 'r') as hf:
            encoded_bytes = hf[isic_id][()]
        image_bgr = cv2.imdecode(encoded_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        from torchvision.transforms.functional import to_pil_image
        image = to_pil_image(image_rgb)
        if self.transform:
            image = self.transform(image)

        return image, meta, label, isic_id

# ========== MODEL ==========
class ResNet50Meta(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(base_model.children())[:-1])
        self.dropout = nn.Dropout(0.5)
        self.meta_fc = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32)
        )
        self.head = nn.Linear(2048 + 32, 1)

    def forward(self, x_img, x_meta):
        x_img = self.cnn(x_img)
        x_img = torch.flatten(x_img, 1)
        x_img = self.dropout(x_img)
        x_meta = self.meta_fc(x_meta)
        x = torch.cat([x_img, x_meta], dim=1)
        return self.head(x)

# ========== LOSS ==========
class SmoothBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing=0.01):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(inputs, targets)

# ========== UTILS ==========
def balance_dataset(df, pos_ratio, seed=42):
    pos_df = df[df['target'] == 1]
    neg_df = df[df['target'] == 0]
    num_pos = len(pos_df)
    num_neg = int((num_pos * (1 - pos_ratio)) / pos_ratio)
    neg_sample = neg_df.sample(n=num_neg, random_state=seed)
    return pd.concat([pos_df, neg_sample]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

# ========== TRANSFORMS ==========
train_transforms = T.Compose([
    T.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ========== TRAINING ==========
def train_fold(fold, train_idx, val_idx, df):
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_dataset = ISIC_HDF5_MetaDataset(train_df, TRAIN_HDF5, transform=train_transforms)
    val_dataset = ISIC_HDF5_MetaDataset(val_df, TRAIN_HDF5, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=48, shuffle=False, num_workers=0)

    model = ResNet50Meta().to(device)
    criterion = SmoothBCEWithLogitsLoss(smoothing=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=6, T_mult=2)

    best_auc = 0
    no_improve_epochs = 0

    for epoch in range(1, 40):
        model.train()
        for x_img, x_meta, y, _ in tqdm(train_loader, desc=f"Fold {fold} | Epoch {epoch}"):
            x_img, x_meta, y = x_img.to(device), x_meta.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                logits = model(x_img, x_meta).view(-1)
                loss = criterion(logits, y)
            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step(epoch)

        model.eval()
        probs, targets = [], []
        with torch.no_grad():
            for x_img, x_meta, y, _ in val_loader:
                x_img, x_meta = x_img.to(device), x_meta.to(device)
                logits = model(x_img, x_meta).view(-1)
                preds = torch.sigmoid(logits).cpu().numpy()
                probs.extend(preds)
                targets.extend(y.numpy())

        auc = roc_auc_score(targets, probs)
        print(f"Fold {fold} | Epoch {epoch} | AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), f"resnet50meta_boosted_fold{fold}.pt")
            print(f"✅ Saved model with AUC {auc:.4f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= PATIENCE:
                print(f"⛔ Early stopping on epoch {epoch} for fold {fold}")
                break

    return best_auc

# ========== MAIN ==========
if __name__ == "__main__":
    df = pd.read_csv(TRAIN_CSV)
    df = balance_dataset(df, pos_ratio=POS_RATIO, seed=SEED)

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['target'])):
        auc = train_fold(fold, train_idx, val_idx, df)
        fold_aucs.append(auc)

    top_folds = np.argsort(fold_aucs)[-TOP_K:][::-1]
    with open("top_folds.txt", "w") as f:
        for fold in top_folds:
            f.write(f"{fold}\n")
    print("\n✅ Top folds saved to top_folds.txt")