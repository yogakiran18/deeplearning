# road-damage-yolo

Automated Road Damage Detection and Spatial Defect Logging using **Ultralytics YOLO11**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ultralytics YOLO11](https://img.shields.io/badge/YOLO-v11-orange.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Project Overview

Municipal road maintenance and highway safety rely heavily on regular pavement inspections. Traditional manual surveys are slow, labor-intensive, hazardous to inspectors, and subjective. **road-damage-yolo** provides an automated deep learning pipeline powered by **Ultralytics YOLO11** to detect, classify, and localize road surface defects in real time from dashcam recordings, drone surveys, or handheld images.

### Target Damage Categories
The detection system classifies four standardized pavement distress categories defined in international road inspection benchmarks (e.g., RDD2020 / RDD2022):

| Class ID | Code | Defect Type | Visual Description | Severity & Impact |
| :---: | :---: | :--- | :--- | :--- |
| `0` | **`D00`** | Longitudinal Crack | Cracks running parallel to the direction of travel | Water infiltration; base erosion |
| `1` | **`D10`** | Transverse Crack | Cracks running perpendicular to traffic flow | Thermal contraction stress |
| `2` | **`D20`** | Alligator Crack | Interconnected polygonal fatigue cracks | Structural fatigue; sub-base failure |
| `3` | **`D40`** | Pothole | Bowl-shaped structural cavities in road surface | High vehicle hazard; tire damage |

### Architecture & Execution Policy
- **Training**: Strictly executed on **Google Colab (GPU)** (`T4`, `A100`). Training is never run locally on low-power hardware.
- **Local Machine**: Dedicated for code editing, data exploration, validation, and local inference testing.
- **Security**: Roboflow API keys are read securely from Google Colab Secrets or local environment variables (`.env`). No credentials are ever hardcoded.

```text
road-damage-yolo/
├── configs/
│   └── data.yaml             # YOLO dataset configuration pointing to splits and classes
├── datasets/
│   ├── road_damage/          # Downloaded Kaggle dataset (train, valid, test splits)
│   └── road_damage_demo/     # Lightweight demo dataset for local smoke tests
├── docs/
│   └── report_outline.md     # Academic project report outline for final-year submissions
├── notebooks/
│   └── colab_train.ipynb     # Complete Google Colab cloud training & backup notebook
├── outputs/
│   ├── road_damage.db        # Relational SQLite database logging inspections & defects
│   └── detections.csv        # Exported detections log
├── samples/                  # Curated real-world pavement damage sample images (D00-D40)
├── src/
│   ├── database.py           # SQLite database manager for inspections & spatial defects
│   ├── prepare_data.py       # Kaggle & Roboflow dataset downloader & split validator
│   ├── train.py              # YOLO11 fine-tuning script with transfer learning and early stopping
│   ├── evaluate.py           # Evaluation script computing mAP and exporting metrics.json
│   ├── predict_image.py      # Image inference script with CSV coordinates export
│   └── predict_video.py      # Video inference script with FPS preservation and HUD overlay
├── tests/                    # Automated unit and smoke tests
├── weights/                  # Local storage for trained checkpoints (best.pt, git-ignored)
├── AGENTS.md                 # Project rules and architectural constraints
├── README.md                 # Primary project documentation
└── requirements.txt          # Python dependencies
```

---

## 2. Dataset

This project supports standard benchmark road damage datasets sourced directly from **Kaggle** (such as RDD2020 / RDD2022 / CRDDC) or **Roboflow Universe**.

### Target Classes
All datasets are mapped and audited against the 4 standard pavement distress classes:
- `0`: **`D00`** (Longitudinal Crack)
- `1`: **`D10`** (Transverse Crack)
- `2`: **`D20`** (Alligator Crack)
- `3`: **`D40`** (Pothole)

---

### Method A: Downloading from Kaggle (Recommended)

#### 1. Obtain Kaggle API Credentials
1. Log in to [Kaggle](https://www.kaggle.com) and go to your **Account Settings** (click your profile avatar -> Settings).
2. Scroll to the **API** section and click **Create New Token**. This downloads `kaggle.json` containing:
   ```json
   {"username": "your_kaggle_username", "key": "your_kaggle_api_key"}
   ```
3. Set your credentials:
   - **Google Colab Secrets (Recommended for GPU Training)**:
     Click the **Key** icon in the Colab sidebar and add two secrets:
     - `KAGGLE_USERNAME`: Your Kaggle username
     - `KAGGLE_KEY`: Your Kaggle API key
   - **Local PowerShell (Windows)**:
     ```powershell
     $env:KAGGLE_USERNAME="your_kaggle_username"
     $env:KAGGLE_KEY="your_kaggle_api_key"
     ```
   - **Local Bash (Linux / macOS)**:
     ```bash
     export KAGGLE_USERNAME="your_kaggle_username"
     export KAGGLE_KEY="your_kaggle_api_key"
     ```
   - *(Optional)* Place `kaggle.json` in `~/.kaggle/kaggle.json` (or `%USERPROFILE%\.kaggle\kaggle.json`).

#### 2. Popular Kaggle Road Damage Datasets
- `aliabdelmenam/rdd-2022` — Multi-national Road Damage Detection 2022 benchmark dataset.
- `chriskue/road-damage-dataset` — RDD2020 road damage dataset.

#### 3. Download and Prepare via CLI
Run `src/prepare_data.py` with `--kaggle`:
```bash
python src/prepare_data.py \
  --kaggle aliabdelmenam/rdd-2022 \
  --out datasets/road_damage
```
**What this automatically handles:**
- Downloads and extracts the dataset archive.
- Flattens nested directories.
- Converts any Pascal VOC XML annotations to normalized YOLO `.txt` labels (`D00, D10, D20, D40`).
- Generates or updates `data.yaml` pointing to train, valid, and test splits.
- Validates 1:1 image-to-label alignment.

---

### Method B: Downloading from Roboflow Universe (Alternative)

#### 1. Getting a Roboflow API Key
1. Sign up for a free account at [roboflow.com](https://roboflow.com).
2. Navigate to **User Settings** -> **Roboflow API** -> Copy your **Private API Key**.
3. Set `ROBOFLOW_API_KEY` in your environment or Colab Secrets:
   ```bash
   export ROBOFLOW_API_KEY="your_private_api_key"
   ```

#### 2. Download via CLI
```bash
python src/prepare_data.py \
  --workspace <workspace> \
  --project <project> \
  --version <version> \
  --out datasets/road_damage
```

---

### Auditing an Existing Dataset
To audit an existing dataset directory without downloading:
```bash
python src/prepare_data.py --verify-only --out datasets/road_damage
```

---

## 3. Local Setup & Quickstart

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/road-damage-yolo.git
cd road-damage-yolo
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Master Pipeline (One-Command Local Run)
To run the entire local pipeline end-to-end (smoke test, data audit, evaluation, image inference, and video inference):
```bash
python run.py
```

Or run individual stages:
```bash
python run.py --mode smoke  # Smoke test only
python run.py --mode eval   # Evaluation only
python run.py --mode infer  # Image & video inference only
```

### 5. Launch the Localhost Web Dashboard (No Colab Needed)
To run the interactive browser interface on localhost for video upload, image inspection, and live webcam detection:
```bash
python app.py
```
*(Or via `python run.py --web`)*

Then open your browser at:
```
http://127.0.0.1:5000
```

**Features of the Local Web Dashboard:**
- **Video Inspection**: Drag & drop any road video or click *"Load Sample Road Video"* to run frame-by-frame YOLO11 detection with live progress tracking, HUD overlay, and automatic H.264 browser playback.
- **Image Analysis**: Upload road photographs or test with bundled samples for side-by-side fault comparisons and coordinate tables.
- **Live Camera / Webcam**: Real-time road damage detection feed directly from your webcam.
- **Pavement Condition Index (PCI)**: Automatic road health rating (Good, Fair, Poor, Critical) based on damage frequency and severity.
- **Export & Download**: One-click download for annotated MP4 videos and detailed CSV defect coordinate logs.

---


## 4. How to Train on Google Colab (GPU)

> [!IMPORTANT]
> Model training must be executed strictly on Google Colab with GPU acceleration. Never run model training locally on CPU.

### Step 1: Open the Notebook in Google Colab
1. Navigate to [Google Colab](https://colab.research.google.com).
2. Go to **File** -> **Upload notebook** and select [`notebooks/colab_train.ipynb`](file:///c:/Users/kiran/OneDrive/Desktop/deep%20learning/notebooks/colab_train.ipynb).
3. Switch runtime to GPU: **Runtime** -> **Change runtime type** -> select **T4 GPU** -> **Save**.

### Step 2: Store your API Key in Colab Secrets
1. Click the **Key icon** (Secrets) in Colab's left sidebar.
2. Add a new secret named `ROBOFLOW_API_KEY` with your private key value.
3. Toggle **Notebook access** to **ON**.

### Step 3: Run the Cloud Pipeline
The notebook executes the following 7 cells in order:
1. `!nvidia-smi`: Verifies active GPU allocation.
2. Clones the repository into `/content/road-damage-yolo` and installs dependencies.
3. Downloads the dataset securely using `ROBOFLOW_API_KEY` via `src/prepare_data.py`.
4. Trains the model using `src/train.py`:
   ```bash
   python src/train.py \
     --data datasets/road_damage/data.yaml \
     --model yolo11s.pt \
     --epochs 50 \
     --imgsz 640 \
     --batch 16 \
     --project runs/train \
     --name road_damage_yolo11s
   ```
5. Evaluates model performance on the test split using `src/evaluate.py`.
6. Mounts Google Drive (`/content/drive`) and backs up `best.pt`, confusion matrices, PR curves, and `metrics.json`.
7. Runs a smoke test on test split images and video frames.

### Step 4: Download Weights Locally
Download `best.pt` from your Google Drive backup (`road_damage_yolo_backup/best.pt`) and place it in the local `weights/` folder as `weights/best.pt`.

---

## 5. Local Inference

Once you have your trained weights in `weights/best.pt` (or using pretrained `yolo11s.pt`), execute local inference:

### A. Image Inference (`src/predict_image.py`)
Run inference on a single image or an entire folder of images:

```bash
# Single image
python src/predict_image.py path/to/pavement.jpg \
  --weights weights/best.pt \
  --conf 0.25 \
  --out outputs/images \
  --csv outputs/detections.csv

# Directory of images
python src/predict_image.py path/to/image_folder/ \
  --weights weights/best.pt \
  --conf 0.25 \
  --out outputs/images \
  --csv outputs/detections.csv
```

**Outputs Produced**:
- **Annotated Images**: Saved to `outputs/images/` with bounding boxes and defect labels.
- **Detections CSV**: Saved to `outputs/detections.csv` with columns:
  `image, class, confidence, x1, y1, x2, y2`

### B. Video Inference (`src/predict_video.py`)
Process vehicle dashcam footage or inspection drone recordings frame-by-frame:

```bash
# Full frame-by-frame processing
python src/predict_video.py path/to/dashcam.mp4 \
  --weights weights/best.pt \
  --conf 0.25 \
  --out outputs/result.mp4

# Accelerated processing (evaluates every 3rd frame, maintaining original FPS and duration)
python src/predict_video.py path/to/dashcam.mp4 \
  --weights weights/best.pt \
  --conf 0.25 \
  --skip 3 \
  --out outputs/result.mp4
```

**Video Features**:
- **Fidelity**: Preserves exact original framerate (FPS), width, and height.
- **Live Progress**: `tqdm` progress bar with running counts per damage class.
- **On-Screen HUD**: Semi-transparent real-time statistics overlay.
- **Summary**: Terminal report displaying total frames processed, total inferences, and class distribution.

### C. Evaluation & Metric Export (`src/evaluate.py`)
Benchmark trained weights against the test split:

```bash
python src/evaluate.py \
  --weights weights/best.pt \
  --data datasets/road_damage/data.yaml \
  --split test \
  --out outputs/metrics.json
```

---

## 6. Results & Benchmark Metrics

> [!NOTE]
> Fill in the placeholders below after training and running `src/evaluate.py`. The metrics are automatically exported to `outputs/metrics.json`.

### Overall Performance (Test Split)

| Model Backbone | Input Resolution (`imgsz`) | Precision (`P`) | Recall (`R`) | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLO11s (Baseline)** | 640 | `[0.XX]` | `[0.XX]` | `[0.XX]` | `[0.XX]` |
| **YOLO11s (High-Res)** | 1024 | `[0.XX]` | `[0.XX]` | `[0.XX]` | `[0.XX]` |

### Per-Class Damage Detection Breakdown

| Class Code | Category Name | Description | Instances | Precision | Recall | AP@0.5 | AP@0.5:0.95 |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`D00`** | Longitudinal Crack | Parallel linear defect | `[XX]` | `[0.XX]` | `[0.XX]` | `[0.XX]` | `[0.XX]` |
| **`D10`** | Transverse Crack | Perpendicular linear defect | `[XX]` | `[0.XX]` | `[0.XX]` | `[0.XX]` | `[0.XX]` |
| **`D20`** | Alligator Crack | Fatigue mesh defect | `[XX]` | `[0.XX]` | `[0.XX]` | `[0.XX]` | `[0.XX]` |
| **`D40`** | Pothole | Road surface cavity | `[XX]` | `[0.XX]` | `[0.XX]` | `[0.XX]` | `[0.XX]` |
| **all** | **Mean Performance** | Overall Average | `[XX]` | `[0.XX]` | `[0.XX]` | `[0.XX]` | `[0.XX]` |

---

## 7. Troubleshooting

### 1. CUDA Out-of-Memory (OOM) on Google Colab
- **Cause**: Batch size is too large for the allocated GPU VRAM when using higher resolutions.
- **Solution**: Decrease batch size:
  ```bash
  python src/train.py --data ... --imgsz 640 --batch 8
  ```
  If training at `--imgsz 1024`, set `--batch 4` or `--batch 8`.

### 2. Missing Test Split in `data.yaml`
- **Cause**: Some Roboflow exports only generate `train` and `valid` splits.
- **Solution**: [src/evaluate.py](file:///c:/Users/kiran/OneDrive/Desktop/deep%20learning/src/evaluate.py) automatically detects if `test` is missing in `data.yaml` and falls back to evaluating on `val`. Alternatively, run:
  ```bash
  python src/evaluate.py --split val
  ```

### 3. Thin Cracks (`D00`, `D10`) Missed by Model
- **Cause**: At 640x640 resolution, narrow hairline cracks lose visual edge definition after downsampling.
- **Solution**: Increase input image resolution to 1024 during training and inference:
  ```bash
  python src/train.py --imgsz 1024 --batch 8
  ```

### 4. Video Codec / MP4 Writer Warnings
- **Cause**: Missing H.264 codecs on minimal Linux or headless environments.
- **Solution**: [src/predict_video.py](file:///c:/Users/kiran/OneDrive/Desktop/deep%20learning/src/predict_video.py) uses the universal `mp4v` FourCC codec. On Linux/Colab, ensure `ffmpeg` is available (`sudo apt-get install ffmpeg`).

### 5. Roboflow Secret Not Found in Colab
- **Cause**: Secret name mismatch or notebook access disabled.
- **Solution**: In Colab's left sidebar, click the **Key icon** -> add `ROBOFLOW_API_KEY` -> enable the **Notebook access** toggle switch.

---

## 8. Academic Project Report
A complete final-year project report outline based on this codebase is available in:
- [`docs/report_outline.md`](file:///c:/Users/kiran/OneDrive/Desktop/deep%20learning/docs/report_outline.md)
