"""
╔══════════════════════════════════════════════════════════════════════════╗
║   Deepfake Image Detection — Full ML Pipeline (OPTIMIZED)              ║
║   Project : Lightweight Real-Time Deepfake Detection                   ║
║   Dataset : FaceForensics++ C23  (Kaggle: xdxd003/ff-c23)             ║
║                                                                          ║
║   OPTIMIZATIONS vs original:                                            ║
║     • HOG + KAZE pre-computed once to disk (was live per epoch)        ║
║     • max_frames_per_class=5000 cap (10k total instead of 500k+)       ║
║     • hog_pixels_per_cell=(16,16) → HOG dim 6,084 vs 26,244           ║
║     • kaze_n_features=32 → KAZE dim 2,048 vs 4,096                    ║
║     • num_workers=4, batch_size=32, persistent_workers=True            ║
║     • epochs=30 + patience=7 (early stop typically at 10-15)           ║
║     • No 50-frame checkpoint overhead during extraction                 ║
║                                                                          ║
║   Pipeline:                                                              ║
║     1. Frame Extraction      (video → jpg frames @ 5 fps)              ║
║     2. Frame Sampling        (stratified 80/10/10 split)               ║
║     3. Image Resizing        (→ 224×224)                               ║
║     4. Normalization         (ImageNet mean/std)                        ║
║     5. Data Augmentation     (flip, blur, brightness, rotate)          ║
║     6. Feature Pre-Cache     (HOG + KAZE saved to disk ONCE)           ║
║     7. MobileNetV2 CNN       (frozen backbone → 1280-d embeddings)     ║
║     8. CNN + HOG + KAZE      (multi-feature fusion)                    ║
║     9. Dense Layer + Dropout                                            ║
║    10. Sigmoid Output        (binary: real=0 / fake=1)                 ║
║    11. Prediction & Eval     (AUC, F1, accuracy, confusion matrix)     ║
║                                                                          ║
║   Kaggle Setup:                                                          ║
║     1. Click "+ Add Data" → search "ff-c23" (by xdxd003) → Add        ║
║     2. Enable GPU (Settings → Accelerator → GPU T4 x2)                 ║
║     3. Run: python deepfake_detection.py                                ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Imports & Warning Suppression
# ══════════════════════════════════════════════════════════════════════════

import os
import cv2
import random
import warnings
import time
import hashlib
import pickle

import numpy as np
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore", message="Error fetching version info",
                        category=UserWarning, module="albumentations")
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print(f"PyTorch  : {torch.__version__}")
print(f"OpenCV   : {cv2.__version__}")
print(f"CUDA     : {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device   : {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Auto-Detect Dataset Root
# ══════════════════════════════════════════════════════════════════════════

def find_dataset_root(base="/kaggle/input") -> Path:
    """
    Scan /kaggle/input for the FF++ dataset mount.
    Handles slug variations: ff-c23, ff-c23-1, etc.
    """
    SIGNAL_NAMES = {
        "real", "fake", "original", "manipulated",
        "deepfakes", "face2face", "faceswap", "neuraltextures",
    }
    IMG_VID = {".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov"}
    base_path = Path(base)

    if not base_path.exists():
        raise FileNotFoundError(f"/kaggle/input not found. Are you running on Kaggle?")

    for sub in sorted(base_path.iterdir()):
        if not sub.is_dir():
            continue
        child_names = {c.name.lower() for c in sub.iterdir()}
        child_exts  = {c.suffix.lower() for c in sub.rglob("*") if c.is_file()}
        if child_names & SIGNAL_NAMES or child_exts & IMG_VID:
            print(f"[Dataset] Found at: {sub}")
            return sub

    first = next(base_path.iterdir(), None)
    if first:
        print(f"[Dataset] No signal found — using first dir: {first}")
        return first

    raise FileNotFoundError(
        f"No dataset found in /kaggle/input.\n"
        f"Fix: Click '+ Add Data' and add xdxd003/ff-c23"
    )


DATASET_ROOT = find_dataset_root()
print(f"\n=== Dataset root: {DATASET_ROOT} ===")
for p in sorted(DATASET_ROOT.iterdir()):
    n = sum(1 for _ in p.rglob("*") if _.is_file())
    print(f"  {p.name}/  ({n} files)")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Global Config
# ══════════════════════════════════════════════════════════════════════════

CONFIG = {
    # Paths
    "dataset_root":   str(DATASET_ROOT),
    "output_frames":  "/kaggle/working/frames",
    "output_dir":     "/kaggle/working",
    "feature_cache":  "/kaggle/working/feat_cache",    # HOG/KAZE cache dir

    # Preprocessing
    "image_size":     224,
    "sample_fps":     5,

    # ── Dataset cap ───────────────────────────────────────────────────
    # FF++ has ~500k+ frames total. 5000 per class = 10k total.
    # Enough to train a solid model. Set None to use everything (slow).
    "max_frames_per_class": 5000,

    # Split
    "train_split":    0.8,
    "val_split":      0.1,
    "test_split":     0.1,

    # Training — optimized values
    "batch_size":     32,          # was 16 → 32 makes better GPU use
    "num_workers":    4,           # was 2 → 4 for Kaggle GPU environment
    "lr":             1e-4,
    "epochs":         30,          # was 100; early stop triggers at ~10-15
    "patience":       7,           # was 10; reduced to match fewer epochs
    "dropout":        0.5,

    # Features — optimised dimensions to cut compute
    "use_hog":              True,
    "use_kaze":             True,
    "hog_orientations":     9,
    "hog_pixels_per_cell":  (16, 16),   # was (8,8) → HOG dim: 6,084 vs 26,244
    "hog_cells_per_block":  (2, 2),
    "kaze_n_features":      32,          # was 64 → KAZE dim: 2,048 vs 4,096

    # Reproducibility
    "seed": 42,
}

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])

os.makedirs(CONFIG["output_frames"], exist_ok=True)
os.makedirs(CONFIG["output_dir"],    exist_ok=True)
os.makedirs(CONFIG["feature_cache"], exist_ok=True)

# Print dimension summary so you know what changed
_sz   = CONFIG["image_size"]
_ppc  = CONFIG["hog_pixels_per_cell"][0]
_cpb  = CONFIG["hog_cells_per_block"][0]
_nblk = (_sz // _ppc) - _cpb + 1
HOG_DIM  = (_nblk ** 2) * (_cpb ** 2) * CONFIG["hog_orientations"]
KAZE_DIM = CONFIG["kaze_n_features"] * 64
FUSED_DIM = 1280 + HOG_DIM + KAZE_DIM
print(f"\n[Config] HOG dim   : {HOG_DIM:,}  (was 26,244 with ppc=8)")
print(f"[Config] KAZE dim  : {KAZE_DIM:,}  (was 4,096 with n=64)")
print(f"[Config] Fused dim : {FUSED_DIM:,}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — Frame Extraction
# Removed: 50-frame checkpoint (unnecessary I/O overhead).
# Added:    per-class frame cap via max_frames_per_class.
# ══════════════════════════════════════════════════════════════════════════

VID_EXTS  = {".mp4", ".avi", ".mov"}
IMG_EXTS  = {".jpg", ".jpeg", ".png"}
REAL_KEYS = {"real", "original", "youtube", "genuine"}
FAKE_KEYS = {"fake", "deepfakes", "face2face", "faceswap",
             "neuraltextures", "manipulated", "altered"}


def label_from_path(p: Path) -> int:
    parts = {part.lower() for part in p.parts}
    if parts & REAL_KEYS: return 0
    if parts & FAKE_KEYS:  return 1
    return -1


def collect_raw_paths(dataset_root: str, max_per_class: int = None):
    """
    Walk dataset_root, collect (path, label) pairs for real/fake.
    Caps each class at max_per_class entries if set.
    """
    root = Path(dataset_root)
    real_paths, fake_paths = [], []

    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext not in (IMG_EXTS | VID_EXTS):
            continue
        lbl = label_from_path(f)
        if   lbl == 0: real_paths.append((str(f), 0))
        elif lbl == 1: fake_paths.append((str(f), 1))

    # Shuffle before capping so we get variety
    random.shuffle(real_paths)
    random.shuffle(fake_paths)

    if max_per_class is not None:
        real_paths = real_paths[:max_per_class]
        fake_paths = fake_paths[:max_per_class]

    result = real_paths + fake_paths
    print(f"[collect] real={len(real_paths)}  fake={len(fake_paths)}  "
          f"total={len(result)}")

    if not real_paths or not fake_paths:
        print("[WARN] One class is empty — first 20 paths found:")
        for p, l in result[:20]:
            print(f"  label={l}  {p}")

    return result


def extract_video_frames(video_path: str, output_dir: str,
                          label: int, fps: int = 5, sz: int = 224):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    native_fps     = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(1, int(native_fps / fps))
    sub     = "real" if label == 0 else "fake"
    out_dir = Path(output_dir) / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_id  = Path(video_path).stem
    saved, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frame = cv2.resize(frame, (sz, sz))
            sp    = out_dir / f"{vid_id}_f{idx:05d}.jpg"
            cv2.imwrite(str(sp), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved.append((str(sp), label))
        idx += 1
    cap.release()
    return saved


def prepare_frames(raw_paths: list, config: dict) -> list:
    """
    If raw_paths contains videos → extract frames to disk.
    If raw_paths contains images → return directly (no extraction needed).
    No periodic checkpointing — it was unnecessary I/O overhead.
    """
    if not raw_paths:
        raise ValueError("raw_paths is empty — check dataset path.")

    sample_ext = Path(raw_paths[0][0]).suffix.lower()
    if sample_ext not in VID_EXTS:
        print(f"[INFO] Pre-extracted images detected ({sample_ext}) — "
              f"skipping extraction.")
        return raw_paths

    all_frames = []
    print(f"[INFO] Extracting frames at {config['sample_fps']} fps ...")
    for vid_path, label in tqdm(raw_paths, desc="Extracting frames"):
        frames = extract_video_frames(
            vid_path, config["output_frames"], label,
            config["sample_fps"], config["image_size"]
        )
        all_frames.extend(frames)

    print(f"[INFO] Total frames extracted: {len(all_frames)}")
    return all_frames


raw_paths  = collect_raw_paths(
    CONFIG["dataset_root"],
    max_per_class=CONFIG["max_frames_per_class"]
)
frame_list = prepare_frames(raw_paths, CONFIG)
print(f"\nSample entry: {frame_list[0]}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 — Train / Val / Test Split (Stratified)
# ══════════════════════════════════════════════════════════════════════════

def split_dataset(frame_list: list, config: dict):
    real = [(p, l) for p, l in frame_list if l == 0]
    fake = [(p, l) for p, l in frame_list if l == 1]
    print(f"[SPLIT INPUT]  real={len(real)}  fake={len(fake)}")

    if not real or not fake:
        raise ValueError("One class is empty — check path labelling.")

    random.shuffle(real)
    random.shuffle(fake)

    def _split(items):
        n   = len(items)
        ntr = max(1, int(n * config["train_split"]))
        nv  = max(1, int(n * config["val_split"]))
        return items[:ntr], items[ntr:ntr + nv], items[ntr + nv:]

    r_tr, r_v, r_te = _split(real)
    f_tr, f_v, f_te = _split(fake)

    train = r_tr + f_tr;  random.shuffle(train)
    val   = r_v  + f_v;   random.shuffle(val)
    test  = r_te + f_te;  random.shuffle(test)

    print(f"[SPLIT OUTPUT] train={len(train)}  val={len(val)}  "
          f"test={len(test)}")
    return train, val, test


train_list, val_list, test_list = split_dataset(frame_list, CONFIG)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 6 — Transforms (Resize + Normalize + Augment)
# ══════════════════════════════════════════════════════════════════════════

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def get_train_transform(sz: int = 224):
    return A.Compose([
        A.Resize(sz, sz),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.4),
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_val_transform(sz: int = 224):
    return A.Compose([
        A.Resize(sz, sz),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


# ══════════════════════════════════════════════════════════════════════════
# SECTION 7 — HOG & KAZE Feature Extractors
# ══════════════════════════════════════════════════════════════════════════

def extract_hog(img_bgr: np.ndarray, config: dict) -> np.ndarray:
    """
    Histogram of Oriented Gradients on grayscale image.
    Returns a 1-D float32 vector.
    Uses ppc=16 by default → 4x smaller vector than ppc=8.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sz   = config["image_size"]
    gray = cv2.resize(gray, (sz, sz))

    ppc  = config["hog_pixels_per_cell"][0]
    cpb  = config["hog_cells_per_block"][0]
    bins = config["hog_orientations"]

    win_size     = (sz, sz)
    block_size   = (cpb * ppc, cpb * ppc)
    block_stride = (ppc, ppc)
    cell_size    = (ppc, ppc)

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride,
                             cell_size, bins)
    return hog.compute(gray).flatten().astype(np.float32)


def extract_kaze(img_bgr: np.ndarray, config: dict) -> np.ndarray:
    """
    KAZE keypoint descriptors (64-D per keypoint).
    Fixed-length output: top-N keypoints by response, zero-padded.
    Uses n=32 by default → 2048-D vs 4096-D with n=64.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kaze = cv2.KAZE_create()
    kps, descs = kaze.detectAndCompute(gray, None)

    n   = config["kaze_n_features"]
    dim = 64

    if descs is None or len(descs) == 0:
        return np.zeros(n * dim, dtype=np.float32)

    order = np.argsort([-kp.response for kp in kps])[:n]
    descs = descs[order]

    if len(descs) < n:
        pad   = np.zeros((n - len(descs), dim), dtype=np.float32)
        descs = np.vstack([descs, pad])

    return descs[:n].flatten().astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 7b — Pre-compute HOG + KAZE to Disk  ← THE KEY SPEED FIX
#
# WHY: In the original code, HOG + KAZE were computed inside __getitem__,
# meaning every image was processed fresh on every epoch.
# For 100k frames × 12 epochs = 1.2 million redundant computations.
# KAZE alone takes ~100ms per image → 12h/epoch on large datasets.
#
# FIX: Compute HOG + KAZE ONCE here, save as .npz files.
# __getitem__ then does a fast np.load() instead of recomputing.
# Pre-compute runs in ~10-30 min (once). Epochs drop to ~5-15 min each.
# ══════════════════════════════════════════════════════════════════════════

FEAT_CACHE_DIR = Path(CONFIG["feature_cache"])
FEAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_key(img_path: str) -> str:
    """Stable MD5-based filename for cache lookup."""
    return hashlib.md5(img_path.encode()).hexdigest()


def precompute_features(frame_list: list, config: dict,
                        cache_dir: Path) -> None:
    """
    Compute HOG + KAZE for every image and save to .npz files.
    Safe to interrupt and resume — already-cached files are skipped.
    """
    cache_dir = Path(cache_dir)
    missing   = [
        (p, l) for p, l in frame_list
        if not (cache_dir / (cache_key(p) + ".npz")).exists()
    ]

    already_done = len(frame_list) - len(missing)
    print(f"[CACHE] {already_done} already cached, "
          f"{len(missing)} to compute ...")

    if not missing:
        print("[CACHE] All features cached. Skipping computation.")
        return

    t0 = time.time()
    for img_path, _ in tqdm(missing, desc="Pre-computing HOG+KAZE"):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        hog_vec  = (extract_hog(img_bgr, config)
                    if config["use_hog"] else np.zeros(1, np.float32))
        kaze_vec = (extract_kaze(img_bgr, config)
                    if config["use_kaze"] else np.zeros(1, np.float32))

        out_path = cache_dir / (cache_key(img_path) + ".npz")
        np.savez_compressed(str(out_path), hog=hog_vec, kaze=kaze_vec)

    elapsed = time.time() - t0
    print(f"[CACHE] Done in {elapsed/60:.1f} min — "
          f"saved to {cache_dir}")


# Run once before training. Safe to interrupt — will resume on re-run.
print("\n[Step 6] Pre-computing HOG + KAZE features ...")
precompute_features(frame_list, CONFIG, FEAT_CACHE_DIR)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 8 — Dataset Class
# HOG + KAZE loaded from .npz cache → fast np.load() not recomputed.
# ══════════════════════════════════════════════════════════════════════════

class FFDataset(Dataset):
    """
    Returns (cnn_tensor, hog_tensor, kaze_tensor, label) per sample.

    cnn_tensor  : float32 [3, H, W]  for MobileNetV2
    hog_tensor  : float32 [HOG_DIM]  loaded from .npz cache
    kaze_tensor : float32 [KAZE_DIM] loaded from .npz cache
    label       : long scalar — 0=real, 1=fake

    HOG/KAZE are NOT recomputed here — they are read from the
    pre-computed cache created in Section 7b.
    """

    def __init__(self, samples: list, transform=None,
                 config: dict = None, cache_dir: Path = None,
                 is_train: bool = True):
        self.samples   = samples
        self.transform = transform
        self.config    = config or CONFIG
        self.cache_dir = Path(cache_dir) if cache_dir else FEAT_CACHE_DIR
        self.is_train  = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read: {path}")

        # CNN branch — albumentations pipeline
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if self.transform:
            img_rgb = self.transform(image=img_rgb)["image"]

        # HOG + KAZE — fast .npz load from pre-computed cache
        npz_path = self.cache_dir / (cache_key(path) + ".npz")
        if npz_path.exists():
            data     = np.load(str(npz_path))
            hog_vec  = data["hog"]
            kaze_vec = data["kaze"]
        else:
            # Fallback: recompute live (runs if precompute was skipped)
            print(f"[WARN] Cache miss for {path} — computing live.")
            hog_vec  = (extract_hog(img_bgr, self.config)
                        if self.config["use_hog"] else np.zeros(1, np.float32))
            kaze_vec = (extract_kaze(img_bgr, self.config)
                        if self.config["use_kaze"] else np.zeros(1, np.float32))

        return (
            img_rgb,
            torch.from_numpy(hog_vec),
            torch.from_numpy(kaze_vec),
            torch.tensor(label, dtype=torch.long),
        )


# ══════════════════════════════════════════════════════════════════════════
# SECTION 9 — Build DataLoaders
# ══════════════════════════════════════════════════════════════════════════

sz = CONFIG["image_size"]

train_ds = FFDataset(train_list, transform=get_train_transform(sz),
                     config=CONFIG, cache_dir=FEAT_CACHE_DIR, is_train=True)
val_ds   = FFDataset(val_list,   transform=get_val_transform(sz),
                     config=CONFIG, cache_dir=FEAT_CACHE_DIR, is_train=False)
test_ds  = FFDataset(test_list,  transform=get_val_transform(sz),
                     config=CONFIG, cache_dir=FEAT_CACHE_DIR, is_train=False)

kw = dict(
    batch_size         = CONFIG["batch_size"],
    num_workers        = CONFIG["num_workers"],
    pin_memory         = torch.cuda.is_available(),
    persistent_workers = True,   # keeps workers alive between epochs → faster
)

train_loader = DataLoader(train_ds, shuffle=True,  **kw)
val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

print(f"\nTrain batches : {len(train_loader)}  ({len(train_ds)} samples)")
print(f"Val   batches : {len(val_loader)}  ({len(val_ds)} samples)")
print(f"Test  batches : {len(test_loader)}  ({len(test_ds)} samples)")

imgs, hogs, kazes, labels = next(iter(train_loader))
print(f"\n── Batch sanity ──")
print(f"  CNN input   : {tuple(imgs.shape)}")
print(f"  HOG vector  : {tuple(hogs.shape)}")
print(f"  KAZE vector : {tuple(kazes.shape)}")
print(f"  Labels      : real={(labels==0).sum().item()}  "
      f"fake={(labels==1).sum().item()}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 10 — Model Definition
#
# Architecture:
#   Branch 1: MobileNetV2 (frozen) → 1280-d CNN embeddings
#   Branch 2: HOG vector → Linear(HOG_DIM, 256) → BN → ReLU
#   Branch 3: KAZE vector → Linear(KAZE_DIM, 256) → BN → ReLU
#   Head:     Concat(1792-d) → Linear → BN → ReLU → Dropout → Linear(1)
# ══════════════════════════════════════════════════════════════════════════

def load_mobilenetv2_offline() -> models.MobileNetV2:
    """
    Load MobileNetV2 weights without internet.
    Kaggle caches them at ~/.cache/torch/hub/checkpoints/.
    If not cached: loads architecture only — enable internet once to cache.
    """
    CACHE_DIR = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    cached    = (list(CACHE_DIR.glob("mobilenet_v2*.pth"))
                 if CACHE_DIR.exists() else [])

    net = models.mobilenet_v2(weights=None)

    if cached:
        weight_file = cached[0]
        print(f"[MobileNetV2] Loading from cache: {weight_file.name}")
        state = torch.load(str(weight_file), map_location="cpu")
        net.load_state_dict(state)
        print("[MobileNetV2] Pretrained ImageNet weights loaded.")
    else:
        print("[MobileNetV2] No cached weights — training from scratch.")
        print("  Enable internet in Kaggle Settings, run once to cache,")
        print("  then disable internet again.")

    return net


class DeepfakeDetector(nn.Module):
    """
    Multi-branch deepfake detector.
    CNN (MobileNetV2 frozen) + HOG projection + KAZE projection → fused head.
    """

    def __init__(self, hog_dim: int, kaze_dim: int, config: dict):
        super().__init__()
        self.config   = config
        self.use_hog  = config["use_hog"]
        self.use_kaze = config["use_kaze"]

        # Branch 1 — MobileNetV2 feature extractor (frozen)
        mobilenet     = load_mobilenetv2_offline()
        self.cnn      = mobilenet.features          # → [B, 1280, 7, 7]
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)     # → [B, 1280, 1, 1]
        self.cnn_dim  = 1280
        for p in self.cnn.parameters():
            p.requires_grad = False                  # frozen

        # Branch 2 — HOG projection
        self.hog_proj_dim = 256
        if self.use_hog:
            self.hog_proj = nn.Sequential(
                nn.Linear(hog_dim, self.hog_proj_dim),
                nn.BatchNorm1d(self.hog_proj_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.hog_proj_dim = 0

        # Branch 3 — KAZE projection
        self.kaze_proj_dim = 256
        if self.use_kaze:
            self.kaze_proj = nn.Sequential(
                nn.Linear(kaze_dim, self.kaze_proj_dim),
                nn.BatchNorm1d(self.kaze_proj_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.kaze_proj_dim = 0

        # Classification head
        fused_dim = self.cnn_dim + self.hog_proj_dim + self.kaze_proj_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(config["dropout"]),
            nn.Linear(512, 1),
        )

    def forward(self, imgs, hog_vecs, kaze_vecs):
        # CNN
        x = self.cnn(imgs)
        x = self.cnn_pool(x)
        x = x.view(x.size(0), -1)          # [B, 1280]
        parts = [x]

        # HOG
        if self.use_hog:
            parts.append(self.hog_proj(hog_vecs))     # [B, 256]

        # KAZE
        if self.use_kaze:
            parts.append(self.kaze_proj(kaze_vecs))   # [B, 256]

        fused  = torch.cat(parts, dim=1)              # [B, fused_dim]
        logits = self.head(fused)                     # [B, 1]
        return logits.squeeze(1)                      # [B]


# Instantiate
model = DeepfakeDetector(
    hog_dim=HOG_DIM, kaze_dim=KAZE_DIM, config=CONFIG
).to(DEVICE)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
print(f"\nModel parameters  : {total_params:,}")
print(f"Trainable params  : {trainable_params:,}  (CNN frozen)")
print(f"Fused feature dim : {FUSED_DIM:,}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 11 — Optimizer, Loss Function, Scheduler
# ══════════════════════════════════════════════════════════════════════════

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG["lr"]
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

print(f"\nOptimizer : Adam  lr={CONFIG['lr']}")
print(f"Loss      : BCEWithLogitsLoss")
print(f"Scheduler : ReduceLROnPlateau (factor=0.5, patience=5)")
print(f"Early stop: patience={CONFIG['patience']}, "
      f"max_epochs={CONFIG['epochs']}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 12 — Training Loop with Early Stopping + Checkpointing
# ══════════════════════════════════════════════════════════════════════════

TRAIN_SAVE_INTERVAL = 5
CKPT_BEST_PATH      = os.path.join(CONFIG["output_dir"], "best_model.pt")
CKPT_INTERVAL_FMT   = os.path.join(CONFIG["output_dir"],
                                    "checkpoint_epoch_{}.pt")
CKPT_RESUME_PATH    = os.path.join(CONFIG["output_dir"], "resume_state.pt")


def save_training_checkpoint(epoch, model, optimizer, scheduler,
                              history, best_val_loss, epochs_no_impv,
                              path: str) -> None:
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "history":         history,
        "best_val_loss":   best_val_loss,
        "epochs_no_impv":  epochs_no_impv,
    }, path)


def load_training_checkpoint(path: str, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    return (
        ckpt["epoch"] + 1,
        ckpt["history"],
        ckpt["best_val_loss"],
        ckpt["epochs_no_impv"],
    )


def run_epoch(loader, model, criterion, optimizer,
              device, is_train: bool):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_preds, all_probs, all_labels = [], [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, hogs, kazes, labels in tqdm(
                loader,
                desc="Train" if is_train else "Val  ",
                leave=False):

            imgs   = imgs.to(device, non_blocking=True)
            hogs   = hogs.to(device, non_blocking=True)
            kazes  = kazes.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            logits = model(imgs, hogs, kazes)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            total_loss += loss.item() * len(labels)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    n        = len(all_labels)
    avg_loss = total_loss / n
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    return avg_loss, acc, f1, auc


# ── Init / Resume ────────────────────────────────────────────────────────

history = {
    "train_loss": [], "val_loss":  [],
    "train_acc":  [], "val_acc":   [],
    "train_f1":   [], "val_f1":    [],
    "train_auc":  [], "val_auc":   [],
}
best_val_loss  = float("inf")
epochs_no_impv = 0
start_epoch    = 1

if os.path.exists(CKPT_RESUME_PATH):
    start_epoch, history, best_val_loss, epochs_no_impv = (
        load_training_checkpoint(CKPT_RESUME_PATH, model,
                                  optimizer, scheduler)
    )
    print(f"[RESUME] From epoch {start_epoch}, "
          f"best val loss: {best_val_loss:.4f}")
else:
    print("[INFO] No resume checkpoint — starting from epoch 1.")


# ── Training loop ────────────────────────────────────────────────────────

print(f"\n{'Epoch':>5}  {'TrLoss':>7}  {'TrAcc':>6}  "
      f"{'TrAUC':>6}  {'VaLoss':>7}  {'VaAcc':>6}  "
      f"{'VaAUC':>6}  {'LR':>8}  {'Time':>6}")
print("─" * 78)

for epoch in range(start_epoch, CONFIG["epochs"] + 1):
    t0 = time.time()

    tr_loss, tr_acc, tr_f1, tr_auc = run_epoch(
        train_loader, model, criterion, optimizer, DEVICE, is_train=True)
    va_loss, va_acc, va_f1, va_auc = run_epoch(
        val_loader, model, criterion, optimizer, DEVICE, is_train=False)

    scheduler.step(va_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    for k, v in zip(
        ["train_loss", "val_loss", "train_acc", "val_acc",
         "train_f1",  "val_f1",  "train_auc", "val_auc"],
        [tr_loss, va_loss, tr_acc, va_acc,
         tr_f1,  va_f1,  tr_auc, va_auc]
    ):
        history[k].append(v)

    flag = ""

    if va_loss < best_val_loss:
        best_val_loss  = va_loss
        epochs_no_impv = 0
        torch.save(model.state_dict(), CKPT_BEST_PATH)
        flag = " ✓"
    else:
        epochs_no_impv += 1

    if epoch % TRAIN_SAVE_INTERVAL == 0:
        interval_path = CKPT_INTERVAL_FMT.format(epoch)
        save_training_checkpoint(epoch, model, optimizer, scheduler,
                                  history, best_val_loss,
                                  epochs_no_impv, interval_path)
        save_training_checkpoint(epoch, model, optimizer, scheduler,
                                  history, best_val_loss,
                                  epochs_no_impv, CKPT_RESUME_PATH)
        flag += f" [ckpt@{epoch}]"

    elapsed = time.time() - t0
    print(f"{epoch:>5}  {tr_loss:>7.4f}  {tr_acc:>6.3f}  "
          f"{tr_auc:>6.3f}  {va_loss:>7.4f}  {va_acc:>6.3f}  "
          f"{va_auc:>6.3f}  {current_lr:>8.2e}  {elapsed:>5.0f}s{flag}")

    if epochs_no_impv >= CONFIG["patience"]:
        print(f"\n[Early Stop] No improvement for "
              f"{CONFIG['patience']} epochs — stopping.")
        save_training_checkpoint(epoch, model, optimizer, scheduler,
                                  history, best_val_loss,
                                  epochs_no_impv, CKPT_RESUME_PATH)
        break

print(f"\nBest val loss : {best_val_loss:.4f} → {CKPT_BEST_PATH}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 13 — Evaluation on Test Set
# ══════════════════════════════════════════════════════════════════════════

print("\n[Eval] Loading best model weights ...")
model.load_state_dict(torch.load(CKPT_BEST_PATH, map_location=DEVICE))
model.eval()

all_preds, all_probs, all_labels = [], [], []

with torch.no_grad():
    for imgs, hogs, kazes, labels in tqdm(test_loader, desc="Testing"):
        imgs   = imgs.to(DEVICE, non_blocking=True)
        hogs   = hogs.to(DEVICE, non_blocking=True)
        kazes  = kazes.to(DEVICE, non_blocking=True)

        logits = model(imgs, hogs, kazes)
        probs  = torch.sigmoid(logits).cpu().numpy()
        preds  = (probs >= 0.5).astype(int)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

acc = accuracy_score(all_labels, all_preds)
f1  = f1_score(all_labels, all_preds, zero_division=0)
try:
    auc = roc_auc_score(all_labels, all_probs)
except ValueError:
    auc = 0.0
cm  = confusion_matrix(all_labels, all_preds)

print(f"\n{'═'*45}")
print(f"  TEST RESULTS")
print(f"{'═'*45}")
print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  F1 Score : {f1:.4f}")
print(f"  AUC-ROC  : {auc:.4f}")
print(f"{'═'*45}")
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
print(f"\nClassification Report:")
print(classification_report(all_labels, all_preds,
                             target_names=["Real", "Fake"],
                             zero_division=0))


# ══════════════════════════════════════════════════════════════════════════
# SECTION 14 — Plot Training Curves
# ══════════════════════════════════════════════════════════════════════════

def plot_history(history: dict, save_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    metrics = [
        ("Loss",     "train_loss", "val_loss"),
        ("Accuracy", "train_acc",  "val_acc"),
        ("AUC-ROC",  "train_auc",  "val_auc"),
    ]

    for ax, (title, tr_key, va_key) in zip(axes, metrics):
        epochs = range(1, len(history[tr_key]) + 1)
        ax.plot(epochs, history[tr_key], "b-o", markersize=3, label="Train")
        ax.plot(epochs, history[va_key], "r-o", markersize=3, label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved to {save_path}")
    plt.close()


plot_history(
    history,
    os.path.join(CONFIG["output_dir"], "training_curves.png")
)

print("\n[Done] Training complete.")
print(f"  Best model : {CKPT_BEST_PATH}")
print(f"  Plots      : {CONFIG['output_dir']}/training_curves.png")
print(f"  Resume     : {CKPT_RESUME_PATH}")
