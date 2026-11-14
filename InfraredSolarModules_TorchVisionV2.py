# -*- coding: utf-8 -*-
"""InfraredSolarModules_TorchVisionV2.py

مشروع تصنيف أعطال الألواح الشمسية — بأسلوب TorchVision v2 الهندسي المتكامل
InfraredSolarModules -> Stratified Split -> Custom SolarDataset -> TorchVision v2 -> WeightedRandomSampler -> Acceptance Tests -> ConvNeXt + EffNetV2 Ensemble
"""

import os
import json
import time
import zipfile
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import v2 as T
from torchvision.utils import make_grid

import timm
from timm.utils import ModelEmaV2
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, balanced_accuracy_score, accuracy_score
)

SEED = 42
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

import argparse

parser = argparse.ArgumentParser(description="InfraredSolarModules TorchVision v2 Training")
parser.add_argument('--epochs', type=int, default=int(os.environ.get("EPOCHS", 3)), help="Number of epochs per model")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
parser.add_argument('--lr', type=float, default=2.5e-4, help="Base learning rate")
args, _ = parser.parse_known_args()

# إعدادات موحّدة قابلة للتعديل من مكان واحد
IMG_H, IMG_W = 160, 96   # نسبة 5:3 مطابقة للأصل (40x24)
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
NUM_WORKERS = 0 if os.name == 'nt' else 2

print(f"Torch: {torch.__version__} | TorchVision: {__import__('torchvision').__version__} | Device: {DEVICE}")

# الخطوة 2 | فك ضغط الأرشيف وتجهيز جدول البيانات
POSSIBLE_ZIPS = [
    "/content/InfraredSolarModules-master.zip",
    "/content/2020-02-14_InfraredSolarModules.zip",
    "/content/drive/MyDrive/InfraredSolarModules-master.zip",
    "/content/drive/MyDrive/2020-02-14_InfraredSolarModules.zip",
    "InfraredSolarModules-master.zip",
    "2020-02-14_InfraredSolarModules.zip",
]

EXTRACT_DIR = Path("solar_data") if os.name == 'nt' else Path("/content/solar_data")
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

def prepare_dataset(extract_dir: Path) -> Path:
    # 1. فحص المجلد المحلي أولاً
    local_meta = list(Path(".").rglob("module_metadata.json"))
    if local_meta:
        return local_meta[0].parent

    # 2. فحص مسار الاستخراج
    existing = list(extract_dir.rglob("module_metadata.json"))
    if existing:
        return existing[0].parent

    # 3. محاولة فك الضغط من الأرشيفات المتاحة
    zip_path = next((Path(p) for p in POSSIBLE_ZIPS if Path(p).exists()), None)
    if zip_path is not None:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
        inner_zips = list(extract_dir.rglob("2020-02-14_InfraredSolarModules.zip"))
        if inner_zips:
            with zipfile.ZipFile(inner_zips[0], "r") as iz:
                iz.extractall(extract_dir)

    meta_files = list(Path(".").rglob("module_metadata.json")) + list(extract_dir.rglob("module_metadata.json"))
    assert len(meta_files) > 0, "❌ لم يتم العثور على module_metadata.json!"
    return meta_files[0].parent

DATA_ROOT = prepare_dataset(EXTRACT_DIR)
print(f"📁 DATA_ROOT: {DATA_ROOT}")
with open(DATA_ROOT / "module_metadata.json", "r") as f:
    raw_meta = json.load(f)

df = pd.DataFrame.from_dict(raw_meta, orient="index").reset_index()
df.rename(columns={"index": "image_id"}, inplace=True)
df["full_path"] = df["image_filepath"].apply(lambda p: str(DATA_ROOT / p))
df = df[df["full_path"].apply(os.path.exists)].reset_index(drop=True)

CLASS_NAMES = sorted(df["anomaly_class"].unique().tolist())
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASS_NAMES)}
df["label"] = df["anomaly_class"].map(CLASS_TO_IDX)

# تقسيم طبقي (80% تدريب / 10% تحقق / 10% اختبار)
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
train_idx, temp_idx = next(sss1.split(df, df["label"]))
train_df = df.iloc[train_idx].reset_index(drop=True)
temp_df = df.iloc[temp_idx].reset_index(drop=True)

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
val_idx, test_idx = next(sss2.split(temp_df, temp_df["label"]))
val_df = temp_df.iloc[val_idx].reset_index(drop=True)
test_df = temp_df.iloc[test_idx].reset_index(drop=True)

print(f"✅ Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,} | Classes: {NUM_CLASSES}")


# الخطوة 3 | معالجة الصورة الحرارية إلى 3 قنوات + Custom Dataset بأسلوب Lazy Loading
def preprocess_thermal_3ch(img_gray: np.ndarray, target_h=IMG_H, target_w=IMG_W) -> np.ndarray:
    """تحويل الصورة الحرارية إلى 3 قنوات: الخام + CLAHE + Fixed-Scale Sobel."""
    resized = cv2.resize(img_gray, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(4, 4))
    ch_clahe = clahe.apply(resized)
    gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad_fixed = np.clip(grad / 160.0 * 255.0, 0, 255).astype(np.uint8)
    return np.stack([resized, ch_clahe, grad_fixed], axis=-1)

class SolarModuleDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, class_names: list, transform=None):
        self.classes = class_names
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.transform = transform

        # نخزّن المسارات والتسميات فقط (Lazy Loading) دون تحميل الصور في الذاكرة
        self.samples = list(zip(dataframe["full_path"].values, dataframe["label"].values))
        self.targets = [int(label) for _, label in self.samples]
        if not self.samples:
            raise ValueError("Dataset is empty!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_3ch = preprocess_thermal_3ch(img_gray)
        image = Image.fromarray(img_3ch)  # PIL Image جاهزة لـ TorchVision v2

        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)

raw_ds = SolarModuleDataset(train_df, CLASS_NAMES, transform=None)
raw_img, raw_lbl = raw_ds[0]
print(f"Raw sample -> Type: {type(raw_img).__name__}, Size (W,H): {raw_img.size}, Label: {raw_ds.classes[raw_lbl]}")


# الخطوة 4 | تحويلات TorchVision v2 للتدريب والتحقق وعكس التطبيع
train_transform = T.Compose([
    T.ToImage(),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.92, 1.08)),
    T.ColorJitter(brightness=0.06, contrast=0.08),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=MEAN, std=STD),
])

eval_transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=MEAN, std=STD),
])

def denormalize(image_chw: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (image_chw.cpu() * std + mean).clamp(0, 1)

# معاينة الفرق: 3 قراءات عشوائية للتدريب مقابل قراءة ثابتة للتحقق
tmp_train = SolarModuleDataset(train_df, CLASS_NAMES, transform=train_transform)
tmp_val = SolarModuleDataset(val_df, CLASS_NAMES, transform=eval_transform)

fig, axes = plt.subplots(1, 4, figsize=(11, 3.5))
for ax in axes[:3]:
    im_t, _ = tmp_train[0]
    ax.imshow(denormalize(im_t).permute(1, 2, 0))
    ax.set_title("Train: random")
    ax.axis("off")

im_v, _ = tmp_val[0]
axes[3].imshow(denormalize(im_v).permute(1, 2, 0))
axes[3].set_title("Val: fixed")
axes[3].axis("off")
plt.suptitle("TorchVision v2: Random Augmentation vs Deterministic Val", fontweight="bold")
plt.tight_layout()
plt.savefig("preview_augmentations.png", dpi=200)
plt.close()
print("Saved preview_augmentations.png")


# الخطوة 5 | تجميع خط البيانات في دالة قابلة لإعادة الاستخدام + اختبارات القبول
def build_solar_pipeline(train_df, val_df, test_df, class_names, batch_size=BATCH_SIZE, balance_train=True):
    train_ds = SolarModuleDataset(train_df, class_names, transform=train_transform)
    val_ds = SolarModuleDataset(val_df, class_names, transform=eval_transform)
    test_ds = SolarModuleDataset(test_df, class_names, transform=eval_transform)

    counts = Counter(train_ds.targets)
    if balance_train:
        # موازنة معتدلة (Power 0.55) لرفع الفئات النادرة مثل Soiling دون إفراط في التكرار
        class_w = {cls_idx: 1.0 / (cnt ** 0.55) for cls_idx, cnt in counts.items()}
        sample_weights = [class_w[y] for y in train_ds.targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_ds), replacement=True,
            generator=torch.Generator().manual_seed(SEED)
        )
        train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'))
    else:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'))

    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'))
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'))
    return train_dl, val_dl, test_dl, counts

train_loader, val_loader, test_loader, train_counts = build_solar_pipeline(
    train_df, val_df, test_df, CLASS_NAMES, batch_size=BATCH_SIZE, balance_train=True
)

# --- اختبارات القبول الصارمة (Acceptance Tests) ---
assert train_loader.dataset.class_to_idx == val_loader.dataset.class_to_idx == test_loader.dataset.class_to_idx
assert set(train_df["image_id"]).isdisjoint(set(val_df["image_id"]))
assert set(train_df["image_id"]).isdisjoint(set(test_df["image_id"]))

x_batch, y_batch = next(iter(train_loader))
assert x_batch.ndim == 4 and x_batch.shape[1:] == (3, IMG_H, IMG_W)
assert x_batch.dtype == torch.float32 and torch.isfinite(x_batch).all()
assert y_batch.ndim == 1 and y_batch.dtype == torch.int64
assert 0 <= int(y_batch.min()) and int(y_batch.max()) < NUM_CLASSES

v1, l1 = val_loader.dataset[0]
v2, l2 = val_loader.dataset[0]
assert torch.equal(v1, v2) and l1 == l2, "Val transform must be deterministic!"
print("✅ Passed all acceptance tests: mapping, no leakage, NCHW, dtype, finite values, val determinism!")

# عرض أول 8 صور من الدفعة باستخدام make_grid
preview = torch.stack([denormalize(im) for im in x_batch[:8]])
grid = make_grid(preview, nrow=4, padding=4)
plt.figure(figsize=(10, 5))
plt.imshow(grid.permute(1, 2, 0))
plt.axis("off")
plt.title("First 8 Balanced Training Images (TorchVision v2 DataLoader)")
plt.tight_layout()
plt.savefig("preview_batch.png", dpi=200)
plt.close()
print("Saved preview_batch.png")


# الخطوة 6 | دالة الخسارة ومحرك التدريب (ModelEmaV2 + AMP)
counts_arr = np.array([train_counts[i] for i in range(NUM_CLASSES)])
loss_weights = (counts_arr.max() / counts_arr) ** 0.25
loss_weights = loss_weights / loss_weights.mean()
LOSS_WEIGHTS_TENSOR = torch.tensor(loss_weights, dtype=torch.float32).to(DEVICE)

class WeightedFocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.0, label_smoothing=0.05):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none",
                             label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()

def train_model_pipeline(arch_name: str, save_path: str, epochs: int = EPOCHS, lr: float = 2.5e-4):
    print(f"\n🚀 تدريب: {arch_name} لمدة {epochs} Epoch على {DEVICE}")
    model = timm.create_model(
        arch_name, pretrained=True, num_classes=NUM_CLASSES,
        drop_rate=0.3, drop_path_rate=0.2
    ).to(DEVICE)
    ema = ModelEmaV2(model, decay=0.995, device=DEVICE)

    criterion = WeightedFocalLoss(weight=LOSS_WEIGHTS_TENSOR, gamma=1.0, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_f1 = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        for b_idx, (imgs, targets) in enumerate(train_loader):
            imgs, targets = imgs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                loss = criterion(model(imgs), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            tr_loss += loss.item() * imgs.size(0)

            if (b_idx + 1) % 50 == 0 or (b_idx + 1) == len(train_loader):
                print(f"  [{arch_name.split('.')[0]}] Ep {epoch:02d}/{epochs:02d} | Batch {b_idx + 1:03d}/{len(train_loader):03d} | Loss: {loss.item():.4f}", flush=True)
        scheduler.step()

        ema.module.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    logits = ema.module(imgs)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_targets.extend(targets.numpy())

        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        val_bal = balanced_accuracy_score(val_targets, val_preds)
        flag = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "model_state_dict": ema.module.state_dict(),
                "arch": arch_name,
                "class_names": CLASS_NAMES,
                "img_size": (IMG_H, IMG_W),
                "val_macro_f1": best_f1
            }, save_path)
            flag = "💾 [BEST]"
        print(f"Epoch {epoch:02d}/{epochs} ({time.time()-t0:4.1f}s) | TrLoss: {tr_loss/len(train_df):.4f} | "
              f"Val BalAcc: {val_bal:.4f} | Val Macro-F1: {val_f1:.4f} {flag}", flush=True)
    return save_path


# الخطوة 7 | التقييم النهائي بالدمج (Ensemble + 4x TTA)
@torch.no_grad()
def get_tta_probs(ckpt_path: str, loader):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    m = timm.create_model(ckpt["arch"], pretrained=False, num_classes=NUM_CLASSES).to(DEVICE)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()

    all_probs, all_targets = [], []
    for imgs, targets in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            p1 = F.softmax(m(imgs), dim=1)
            p2 = F.softmax(m(torch.flip(imgs, dims=[3])), dim=1)
            p3 = F.softmax(m(torch.flip(imgs, dims=[2])), dim=1)
            p4 = F.softmax(m(torch.flip(imgs, dims=[2, 3])), dim=1)
            probs = (p1 + p2 + p3 + p4) / 4.0
        all_probs.append(probs.cpu().numpy())
        all_targets.extend(targets.numpy())
    return np.vstack(all_probs), np.array(all_targets)


if __name__ == '__main__':
    CKPT_CONVNEXT = train_model_pipeline("convnext_tiny.fb_in22k_ft_in1k", "best_convnext.pth", epochs=EPOCHS, lr=2.5e-4)
    CKPT_EFFNET = train_model_pipeline("tf_efficientnetv2_s.in21k_ft_in1k", "best_effnetv2s.pth", epochs=EPOCHS, lr=3.5e-4)

    print("\n🔍 جاري تقييم النماذج باستخدام Ensemble + 4x TTA...")
    probs_conv, test_targets = get_tta_probs(CKPT_CONVNEXT, test_loader)
    probs_eff, _ = get_tta_probs(CKPT_EFFNET, test_loader)
    probs_ens = 0.55 * probs_conv + 0.45 * probs_eff

    for name, p in [("ConvNeXt-Tiny", probs_conv),
                    ("EfficientNetV2-S", probs_eff),
                    ("🏆 Ensemble (55% ConvNeXt + 45% EffNetV2)", probs_ens)]:
        preds = p.argmax(axis=1)
        print(f"{name:<42} | Acc: {accuracy_score(test_targets, preds)*100:.2f}% | "
              f"BalAcc: {balanced_accuracy_score(test_targets, preds)*100:.2f}% | "
              f"Macro-F1: {f1_score(test_targets, preds, average='macro', zero_division=0)*100:.2f}%")

    ens_preds = probs_ens.argmax(axis=1)
    print("\n📋 التقرير التفصيلي:")
    print(classification_report(test_targets, ens_preds, target_names=CLASS_NAMES, digits=4, zero_division=0))

    cm = confusion_matrix(test_targets, ens_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title("Normalized Confusion Matrix — TorchVision v2 + Ensemble", fontweight="bold")
    ax.set_ylabel("True Class")
    ax.set_xlabel("Predicted Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("confusion_matrix_ensemble.png", dpi=300)
    plt.close()
    print("Saved confusion_matrix_ensemble.png")

