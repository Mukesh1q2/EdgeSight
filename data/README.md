# EdgeSight Data Pipeline

This directory contains scripts for acquiring and preprocessing fall detection datasets.

## Overview

The data pipeline consists of three stages:
1. **Download**: Acquire UR Fall and Le2i datasets
2. **Preprocess**: Extract pose keypoints and create temporal windows
3. **Dataset**: PyTorch Dataset with stratified splits

## Directory Structure

```
data/
├── raw/                    # Original video files
│   ├── urfall/
│   └── le2i/
├── processed/              # Numpy arrays
│   ├── X.npy              # (N, 30, 51) features
│   └── y.npy              # (N,) labels
├── download_datasets.py
├── preprocess.py
└── dataset.py
```

## Usage

### 1. Download Datasets

```bash
python data/download_datasets.py
```

**Note**: UR Fall requires manual download due to academic registration requirements.
- Visit: http://fenix.ur.edu.pl/mkepski/ds/uf.html
- Register and download
- Extract to `data/raw/urfall/`

Le2i can be downloaded automatically via KaggleHub if available, or manually from:
- https://www.kaggle.com/datasets/muhammadwaseem18/le2i-fall-dataset

### 2. Preprocess Videos

```bash
python data/preprocess.py
```

This will:
- Extract 17 pose keypoints per frame using MediaPipe
- Create overlapping 30-frame windows (stride=15)
- Save as `X.npy` (shape: N, 30, 51) and `y.npy` (shape: N,)

### 3. Test Dataset

```bash
python data/dataset.py
```

Validates the PyTorch Dataset class and creates train/val/test splits.

## Data Format

### Input Features (X)
- Shape: `(N, 30, 51)`
  - N: Number of clips
  - 30: Frames per clip
  - 51: Keypoint features (17 keypoints × 3 values)

Each frame contains 17 MediaPipe pose landmarks:
```
[x1, y1, c1, x2, y2, c2, ..., x17, y17, c17]
```
Where x, y are normalized coordinates [0, 1] and c is confidence.

### Keypoints (17 landmarks)
1. Nose
2. Left shoulder
3. Right shoulder
4. Left elbow
5. Right elbow
6. Left wrist
7. Right wrist
8. Left hip
9. Right hip
10. Left knee
11. Right knee
12. Left ankle
13. Right ankle
14. Left foot index
15. Right foot index

### Labels (y)
- 1: Fall detected
- 0: Normal activity

## Expected Statistics

After preprocessing, you should see:
- Total clips: 2,000-5,000+ (depends on datasets)
- Fall ratio: 15-30% (class imbalance is expected)
- X.npy shape: (N, 30, 51)
- y.npy shape: (N,)

## Troubleshooting

### No pose detected in videos
- Check video quality and lighting
- MediaPipe may fail on blurry or dark videos
- Consider increasing `min_detection_confidence`

### Low fall ratio
- This is normal - falls are rare events
- The training script handles imbalance via `pos_weight`
- Consider data augmentation if ratio < 10%

### Out of memory during preprocessing
- Process datasets separately:
  ```bash
  python data/preprocess.py --raw-dir data/raw/urfall --output-dir data/processed/ur
  python data/preprocess.py --raw-dir data/raw/le2i --output-dir data/processed/le2i
  ```
- Reduce `window_size` or process in smaller batches
