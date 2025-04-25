import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import h5py
import cv2
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torchvision.transforms.functional import to_pil_image
import os

# ----------------------- Config -----------------------
TTA = 10
BATCH_SIZE = 16
TOP_FOLDS = [0, 1, 2]  # Most consistent high-AUC folds
MODEL_TEMPLATE = "resnet50meta_boosted_fold{}.pt"
CSV_PATH = "/Users/santiagovillasenor/Library/CloudStorage/Dropbox-HeuristicsFinansoft/Jaime Villasenor/Personal/Santi/ITAM/8_Semestre/Neural Networks/Kaggle/KaggleSubmision/data/students-test-metadata.csv"
HDF5_PATH = "/Users/santiagovillasenor/Library/CloudStorage/Dropbox-HeuristicsFinansoft/Jaime Villasenor/Personal/Santi/ITAM/8_Semestre/Neural Networks/Kaggle/KaggleSubmision/data/test-image.hdf5"
SUBMISSION_PATH = "/Users/santiagovillasenor/Library/CloudStorage/Dropbox-HeuristicsFinansoft/Jaime Villasenor/Personal/Santi/ITAM/8_Semestre/Neural Networks/Kaggle/KaggleSubmision/testcsv/submission80_best4_avg1.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------- Dataset -----------------------
class ISICMetaTTADataset(Dataset):
    def __init__(self, df, hdf5_path, transform=None, tta=1):
        self.df = df.reset_index(drop=True)
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.tta = tta
        self.meta = self._process_metadata(df)

    def _process_metadata(self, df):
        df = df.copy()
        site_col = 'anatom_site_general'
        df['sex'] = LabelEncoder().fit_transform(df['sex'].fillna("unknown"))
        df[site_col] = LabelEncoder().fit_transform(df[site_col].fillna("unknown"))
        df['age_approx'] = df['age_approx'].fillna(df['age_approx'].median())
        meta = df[['age_approx', 'sex', site_col]].values
        return torch.tensor(StandardScaler().fit_transform(meta), dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        isic_id = row['isic_id']
        meta = self.meta[idx]

        with h5py.File(self.hdf5_path, 'r') as hf:
            encoded_bytes = hf[isic_id][()]
        image_bgr = cv2.imdecode(encoded_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = to_pil_image(image_rgb)
        images = [self.transform(image) for _ in range(self.tta)]
        return torch.stack(images), meta, isic_id

# ----------------------- Transforms -----------------------
tta_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ----------------------- Model -----------------------
class ResNet50Meta(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(base_model.children())[:-1])
        self.dropout = nn.Dropout(0.2)
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

# ----------------------- Load Data -----------------------
df = pd.read_csv(CSV_PATH)
dataset = ISICMetaTTADataset(df, HDF5_PATH, transform=tta_transform, tta=TTA)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ----------------------- Ensemble Prediction -----------------------
all_fold_probs = []

for fold in TOP_FOLDS:
    model = ResNet50Meta().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_TEMPLATE.format(fold), map_location=DEVICE))
    model.eval()

    fold_probs = []
    with torch.no_grad():
        for image_stack, meta, _ in tqdm(dataloader, desc=f"Fold {fold} Inference"):
            B, T, C, H, W = image_stack.shape
            image_stack = image_stack.view(-1, C, H, W).to(DEVICE)
            meta = meta.to(DEVICE).repeat_interleave(T, dim=0)
            logits = model(image_stack, meta).view(B, T)
            probs = torch.sigmoid(logits)
            avg_probs = probs.mean(dim=1).cpu().numpy()
            fold_probs.extend(avg_probs)

    all_fold_probs.append(fold_probs)

# ----------------------- Average Across Folds -----------------------
final_probs = np.mean(np.stack(all_fold_probs, axis=0), axis=0)

# ----------------------- Save Submission -----------------------
pred_df = pd.DataFrame({
    "isic_id": df["isic_id"],
    "target": final_probs
})
pred_df = pred_df.sort_values("isic_id").reset_index(drop=True)
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True) if os.path.dirname(SUBMISSION_PATH) else None
pred_df.to_csv(SUBMISSION_PATH, index=False)
print(f"✅ Submission saved to {SUBMISSION_PATH}")
