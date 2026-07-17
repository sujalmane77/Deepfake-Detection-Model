# Lightweight Real-Time Deepfake Image Detection using Efficient CNN Models

> B.Tech Information Technology — Machine Learning Algorithms Project  
> NMIMS MPSTME, Mumbai

---

## What This Project Does

Detects whether a facial image is **real or AI-generated (deepfake)** using a multi-feature CNN pipeline. The model fuses three types of features — CNN spatial embeddings, HOG texture descriptors, and KAZE keypoint descriptors — to classify faces extracted from the FaceForensics++ dataset.

The core focus is **speed + accuracy**: the model is designed to run in real-time on standard hardware without requiring a high-end GPU for inference.

---

## Project Structure

```
project/
├── deepfake_detection.py       ← complete training + evaluation pipeline
├── README.md                   ← this file
│
├── /kaggle/working/            ← all outputs written here during training
│   ├── frames/
│   │   ├── real/               ← extracted real face frames (.jpg)
│   │   └── fake/               ← extracted fake face frames (.jpg)
│   ├── feat_cache/             ← pre-computed HOG + KAZE features (.npz)
│   ├── best_model.pt           ← best model weights (lowest val loss)
│   ├── resume_state.pt         ← full checkpoint for resuming training
│   ├── checkpoint_epoch_N.pt   ← periodic checkpoints every 5 epochs
│   └── training_curves.png     ← loss / accuracy / AUC plots
```

---

## Dataset

**FaceForensics++ C23** — the standard benchmark for deepfake detection research.

| Detail | Info |
|---|---|
| Kaggle slug | `xdxd003/ff-c23` |
| GitHub | github.com/ondyari/FaceForensics |
| Total size | ~500k+ frames from 1,000 YouTube videos |
| Classes | Real (pristine) and Fake (manipulated) |
| Manipulation methods | Face2Face, FaceSwap, DeepFakes, NeuralTextures |
| Compression | H.264 CRF 23 (high quality, realistic social-media scenario) |
| Used in this project | 10,000 frames (5,000 real + 5,000 fake) |

To add the dataset on Kaggle:
1. Open your notebook
2. Click **+ Add Data** (top right)
3. Search `ff-c23` → select dataset by `xdxd003` → click **Add**

---

## Architecture

```
Input Image (224×224)
        │
        ├─────────────────────┬──────────────────────┐
        ▼                     ▼                      ▼
  MobileNetV2            HOG Extractor          KAZE Extractor
  (frozen backbone)      ppc=16, bins=9         top-32 keypoints
  AdaptiveAvgPool        6,084-D vector         2,048-D vector
  1,280-D embedding           │                      │
        │                     ▼                      ▼
        │              Linear(6084→256)       Linear(2048→256)
        │              BatchNorm + ReLU       BatchNorm + ReLU
        │                     │                      │
        └─────────────────────┴──────────────────────┘
                              │
                   Concatenate → 9,412-D
                              │
                    Linear(9412→512)
                    BatchNorm + ReLU
                    Dropout(0.5)
                    Linear(512→1)
                    Sigmoid → Real / Fake
```

**Why three branches?**
- CNN captures high-level spatial artifacts introduced by GAN generators
- HOG captures texture gradient patterns that deepfakes distort subtly
- KAZE captures local structural keypoints that face-swap algorithms misalign

---

## How to Run

### On Kaggle (recommended)

1. Go to [kaggle.com](https://kaggle.com) → **New Notebook**
2. Settings → Accelerator → **GPU T4 x2**
3. Add the dataset (see above)
4. Upload `deepfake_detection.py` or paste the contents into cells
5. Run all — the script handles everything automatically:
   - Auto-detects dataset location under `/kaggle/input/`
   - Extracts frames from video (or uses pre-extracted images directly)
   - Pre-computes all HOG + KAZE features to disk cache (runs **once**)
   - Trains with early stopping
   - Evaluates on test set
   - Saves model, plots, and checkpoints

### Locally

```bash
# Install dependencies
pip install torch torchvision opencv-python albumentations scikit-learn tqdm matplotlib

# Run
python deepfake_detection.py
```

Make sure to update `DATASET_ROOT` inside the script if running locally (the `find_dataset_root()` function looks under `/kaggle/input` by default).

---

## Configuration

All settings live in the `CONFIG` dict at the top of the script. Key ones to know:

| Setting | Default | What it controls |
|---|---|---|
| `max_frames_per_class` | `5000` | Frames per class used for training. Set `None` to use all (slow). |
| `batch_size` | `32` | Increase to `64` if you have a large GPU. |
| `num_workers` | `4` | Parallel data loading threads. Match to your CPU core count. |
| `epochs` | `30` | Max epochs. Early stopping usually triggers around 10-15. |
| `patience` | `7` | Early stop if val loss doesn't improve for 7 epochs. |
| `lr` | `1e-4` | Learning rate for Adam optimizer. |
| `hog_pixels_per_cell` | `(16, 16)` | Smaller = bigger HOG vector = slower. `(8,8)` gives 26,244-D. |
| `kaze_n_features` | `32` | Number of KAZE keypoints. Higher = more detail, slower pre-compute. |

---

## Training Timeline (Kaggle GPU T4)

| Step | Time |
|---|---|
| Frame extraction (if video input) | ~5 min |
| HOG + KAZE pre-computation (runs once, then cached) | ~15–25 min |
| Per training epoch | ~8–12 min |
| Full training (early stop ~epoch 12) | ~2–3 hours total |

> **If you interrupt training**, just re-run the script. The feature cache is reused instantly (skips pre-computation), and training resumes from `resume_state.pt`.

---

## Expected Results

| Metric | Target | XceptionNet Baseline (FF++) |
|---|---|---|
| Accuracy (C23 HQ) | > 88% | 95.73% |
| AUC-ROC | > 0.95 | ~0.99 |
| F1 Score | > 0.90 | ~0.99 |
| Model size | < 15 MB | ~87 MB |
| Inference speed | < 50 ms/frame | ~300 ms/frame |

The accuracy gap vs XceptionNet exists because this model is ~6× smaller, trains in hours instead of days, and runs in real-time without a GPU — which is the whole point.

---

## Key Optimisations (Why the Original Took 12 Hours per Epoch)

The original notebook computed HOG and KAZE **live inside `__getitem__`**, meaning every image was re-processed from scratch on every epoch. KAZE alone takes ~100ms per image on CPU. With 100,000 frames that's nearly 3 hours just for feature extraction per epoch — before any actual training.

| Problem | Fix | Effect |
|---|---|---|
| KAZE recomputed every epoch | Pre-compute once → `.npz` cache | Biggest fix: 12hr → ~10 min/epoch |
| HOG recomputed every epoch | Same cache | Eliminates all live HOG computation |
| HOG vector 26,244-D (ppc=8) | Changed to ppc=16 → 6,084-D | 4× smaller, faster Linear layer |
| KAZE vector 4,096-D (n=64) | Changed to n=32 → 2,048-D | 2× smaller |
| Training on 500k+ frames | Cap at 5,000 per class | 50× less data per epoch |
| 2 workers, batch size 16 | 4 workers, batch 32, persistent | Better GPU utilisation |
| 100 epochs | 30 epochs + patience=7 | Early stop at ~epoch 12 in practice |
| 50-frame extraction checkpoint | Removed | Unnecessary I/O during extraction |

---

## Dependencies

```
torch >= 1.13
torchvision >= 0.14
opencv-python >= 4.7
albumentations >= 1.3
scikit-learn >= 1.2
tqdm
matplotlib
numpy
```

All pre-installed on Kaggle GPU notebooks.

---

## Resuming Interrupted Training

The script saves `resume_state.pt` automatically every 5 epochs and on early stop. To resume:

```python
# Just re-run the script — it detects resume_state.pt automatically
python deepfake_detection.py
```

The feature cache is also persistent — it won't recompute features it has already saved.

---

## References

1. Rossler et al., *FaceForensics++: Learning to Detect Manipulated Facial Images*, ICCV 2019. [arXiv:1901.08971](https://arxiv.org/abs/1901.08971)
2. Chollet, *Xception: Deep Learning with Depthwise Separable Convolutions*, CVPR 2017. [arXiv:1708.04896](https://arxiv.org/abs/1708.04896)
3. Afchar et al., *MesoNet: A Compact Facial Video Forgery Detection Network*, WIFS 2018. [arXiv:1809.00888](https://arxiv.org/abs/1809.00888)
4. Bonettini et al., *Video Face Manipulation Detection Through Ensemble of CNNs*, ICPR 2020.
5. Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
6. FaceForensics++ Dataset — [github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)

---

*Project for Machine Learning Algorithms course — B.Tech IT, NMIMS MPSTME*
