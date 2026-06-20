# Flood Detection with Prithvi-EO-2.0 — AISEHack Edition 1

Fine-tuning the **Prithvi-EO-2.0 (600M)** geospatial foundation model for binary flood segmentation on multi-sensor satellite imagery. Built for **AISEHack Edition 1** (Theme 1: Flood Detection), a national AI hackathon organized by the **Anusandhan National Research Foundation (ANRF)**, co-organized with **IBM** and held at **IIIT Hyderabad** (April 4–5, 2026).

**Final leaderboard score: 0.714 mIoU**

---

## About AISEHack

AISEHack is a national AI for Science and Engineering Hackathon organized by the Anusandhan National Research Foundation (ANRF). Edition 1 was co-organized with IBM and IIT Delhi, held at IIIT Hyderabad in April 2026, with a prize pool of ₹7,00,000 and 50+ finalist teams. Theme 1 focused on remote sensing and geospatial AI for detecting flood-prone regions using satellite data.

---

## Task

Binary semantic segmentation of flooded regions from 6-channel satellite imagery:

| Channel | Modality | Band |
|---|---|---|
| 1 | SAR | HH |
| 2 | SAR | HV |
| 3 | Optical | Green |
| 4 | Optical | Red |
| 5 | Optical | NIR |
| 6 | Optical | SWIR |

Output: Per-pixel binary mask (0 = non-flood, 1 = flood)

---

## Model Architecture

- **Backbone**: `prithvi_eo_v2_600_tl` — NASA/IBM Prithvi-EO-2.0 (600M parameter ViT)
- **Decoder**: UperNet (`decoder_channels=256`, `scale_modules=True`)
- **Neck**: `ReshapeTokensToImage` to bridge ViT token output to spatial feature maps
- **Framework**: [TerraTorch](https://github.com/IBM/terratorch) + PyTorch Lightning

### Partial Freezing Strategy

Rather than freezing the full backbone or fine-tuning everything, a targeted partial-freeze was used:

1. Freeze all parameters
2. Unfreeze: UperNet decoder, neck, and classification head
3. Unfreeze: Last 4 ViT blocks (`blocks.28–31`) + final LayerNorm

This kept the majority of Prithvi's pretrained geospatial representations frozen while allowing the top-level features and decoder to adapt to the flood detection task.

**Trainable parameters**: ~X% of total (decoder + last 4 blocks)

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Epochs | 35 |
| Learning Rate | 2e-5 |
| Optimizer | AdamW |
| Weight Decay | 0.1 |
| Loss | Dice (IoU-optimized) |
| Precision | fp16-mixed |
| Batch Size | 2 |
| Head Dropout | 0.1 |
| Class Weights | [0.26, 0.74] (flood-weighted) |
| Seed | 42 |
| GPU | Kaggle T4 |

### Data Augmentation

```python
A.HorizontalFlip(p=0.4)
A.VerticalFlip(p=0.7)
A.RandomRotate90(p=0.6)
A.Transpose(p=0.6)
```

### Band Normalization (Z-score, per-band)

| Band | Mean | Std |
|---|---|---|
| HH (SAR) | 841.13 | 473.71 |
| HV (SAR) | 371.62 | 170.36 |
| Green | 1734.18 | 623.07 |
| Red | 1588.31 | 612.88 |
| NIR | 1742.85 | 564.58 |
| SWIR | 1218.56 | 528.09 |

---

## Results

| Metric | Score |
|---|---|
| **mIoU (final leaderboard)** | **0.714** |
| Best val checkpoint | `epoch=22`, val mIoU=0.6947 |

<img width="1990" height="590" alt="image" src="https://github.com/user-attachments/assets/d34314c4-36f7-4b3a-bd67-b409ab244bad" />
<img width="1990" height="590" alt="image" src="https://github.com/user-attachments/assets/3828e730-0fc3-4bd5-8116-1a0aa5d740e2" />
<img width="1990" height="590" alt="image" src="https://github.com/user-attachments/assets/7fc6bf55-5bcb-4fb4-9ea5-375c857d9ca4" />



Visualization uses NIR-Red-Green false color composite (bands 5-4-3) with a red overlay on predicted flood pixels (α=0.4 blend).

---

## Key Engineering Challenges

### fp16 NaN Loss
Mixed precision training caused NaN loss early in training. Root cause: GroupNorm layers accumulating NaN in fp16. Fixed by ensuring GroupNorm operates in fp32 or by switching to Dice loss which is more numerically stable than Focal loss in fp16.

### BatchNorm Eval-Mode Failure
BatchNorm running stats (mean/var computed during training) produced incorrect predictions at inference time when called in `eval()` mode. Fixed by forcing `train()` mode during test inference to use batch statistics rather than stale running stats.

### Boundary Loss Overflow
Boundary-aware loss terms overflowed in fp16 due to large gradient magnitudes at mask edges. Resolved by removing boundary loss and relying solely on Dice loss for the final submission.

### Tiled Inference (explored, not used in final)
Experimented with tiled inference (512×512 crops, 256 stride, patch averaging) to handle large image sizes, but it was not included in the final submission.

---

## Submission Format

Predictions are exported as GeoTIFF (int16, LZW compressed, nodata=-1) and converted to RLE (column-major) for Kaggle submission.

```python
def mask_to_rle(mask):
    pixels = mask.flatten(order="F")  # column-major
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)
```

---

## Environment

```bash
pip install terratorch "numpy<2.0.0"
```

| Library | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.x |
| TerraTorch | latest |
| PyTorch Lightning | latest |
| Rasterio | latest |
| Albumentations | latest |
| NumPy | <2.0.0 |

---

## Repo Structure

```
├── Prithvi_FT.ipynb       # Main training + inference notebook
├── README.md
```

---

## References

- [Prithvi-EO-2.0](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M) — IBM/NASA geospatial foundation model
- [TerraTorch](https://github.com/IBM/terratorch) — Fine-tuning framework for geospatial models
- [AISEHack](https://precog.iiit.ac.in/aisehack) — ANRF national hackathon
- [Kaggle Competition](https://www.kaggle.com/competitions/aisehack-theme-1) — `aisehack-theme-1`
