"""YOLO11 Road Damage Detection Fine-Tuning Script.

This script fine-tunes an Ultralytics YOLO11 model on a road damage detection
dataset (classes: D00, D10, D20, D40) using transfer learning.

EXECUTION POLICY REMINDER:
- Full production training must be executed on Google Colab (GPU).
- Local training is supported for rapid dry-runs and sanity verification.

Hyperparameter Rationale & Tuning Guide:
----------------------------------------
1. Model Architecture & Weights (--model, default: 'yolo11s.pt'):
   - Choice: YOLO11 small (`yolo11s.pt`) initialized with COCO pretrained weights.
   - Why: Transfer learning leverages low-level visual features (edges, textures)
     learned from massive datasets, speeding up convergence on road crack patterns.
     `yolo11s` balances parameter count (~9.4M) and inference latency with sufficient
     feature representation capacity to distinguish complex crack textures.
   - When to change:
     * Switch to `yolo11n.pt` (nano) for resource-constrained edge/embedded deployment.
     * Switch to `yolo11m.pt` or `yolo11l.pt` if validation mAP50-95 plateaus and higher
       GPU VRAM (e.g. Colab A100 or V100) is accessible.

2. Input Image Size (--imgsz, default: 640):
   - Choice: 640x640 pixels.
   - Why: Standard YOLO resolution providing a well-tested trade-off between
     receptive field resolution, training speed, and GPU memory footprint.
   - When to change:
     * Increase to `--imgsz 1024` or `1280` when detecting narrow, hairline cracks
       (D00 longitudinal, D10 transverse) or distant potholes (D40) where downsampling
       at 640 causes thin crack boundaries to vanish. (Note: larger imgsz increases VRAM usage).
     * Decrease to `--imgsz 512` only if experiencing severe GPU memory constraints.

3. Epochs (--epochs, default: 50):
   - Choice: 50 epochs.
   - Why: Fine-tuning a pretrained model on curated datasets typically converges
     between 30 to 50 epochs.
   - When to change: Increase to 100 if training loss continues to descend without
     validation loss divergence.

4. Early Stopping Patience (patience=15):
   - Choice: 15 epochs without validation metric improvement (mAP50-95).
   - Why: Prevents overfitting to the training split and conserves Colab GPU compute units.
   - When to change:
     * Increase to 20-25 if learning rate warmup or decay schedules cause delayed gains.
     * Decrease to 10 for quick exploratory iterations.

5. Batch Size (--batch, default: 16):
   - Choice: 16 images per mini-batch.
   - Why: Provides stable batch normalization statistics while fitting comfortably within
     standard Colab GPU memory (Tesla T4 with ~15GB VRAM) at 640x640 resolution.
   - When to change:
     * Reduce to 8 or 4 if using `--imgsz 1024` or heavier models (`yolo11m`/`yolo11l`) to
       prevent CUDA Out-Of-Memory (OOM) errors.
     * Increase to 32 or 64 on larger GPUs (A100) to accelerate throughput.

6. Vertical Flip Augmentation (flipud=0.0):
   - Choice: Disabled (0.0 probability).
   - Why: Road damage images are captured by forward-facing or downward-facing cameras
     with a fixed gravity orientation. Asphalt surfaces and horizon lines are never upside down;
     inverting images vertically generates unrealistic artifacts that degrade model localization.

7. Horizontal Flip Augmentation (fliplr=0.5):
   - Choice: 50% probability.
   - Why: Road cracks, lanes, and asphalt defects are horizontally symmetrical. Flipping
     left-to-right provides realistic data variation without violating domain physics.

8. Training Plots (plots=True):
   - Choice: Enabled.
   - Why: Generates PR curves, F1-confidence curves, confusion matrices, and validation
     prediction batches in the experiment run directory for quantitative evaluation.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Opt out of MLflow local file-store exception if MLflow is installed in Python environment
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import torch
from ultralytics import YOLO, settings

# Disable MLflow integration callback to prevent local filesystem tracking conflicts
try:
    settings.update({"mlflow": False})
except Exception:
    pass


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for YOLO11 training."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Ultralytics YOLO11 on road damage dataset (D00, D10, D20, D40)."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml dataset specification file (e.g., 'datasets/road_damage/data.yaml').",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s.pt",
        help="Pretrained YOLO11 model checkpoint for transfer learning (default: 'yolo11s.pt').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Total number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution in pixels (default: 640; use 1024 for thin cracks).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size per forward/backward pass (default: 16; reduce if OOM occurs).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Project directory where training runs and artifacts are saved (default: 'runs/train').",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="road_damage_yolo11s",
        help="Experiment name within the project directory (default: 'road_damage_yolo11s').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on ('0', '0,1', 'cpu'). Defaults to GPU if available.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 2),
        help="Number of worker threads for dataloading.",
    )
    parser.add_argument(
        "--exist_ok",
        action="store_true",
        default=True,
        help="Overwrite existing experiment folder instead of creating road_damage_yolo11s2 (default: True).",
    )
    return parser.parse_args()


def train_yolo(
    data: str,
    model: str = "yolo11s.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/train",
    name: str = "road_damage_yolo11s",
    device: str | None = None,
    workers: int = 2,
    exist_ok: bool = True,
) -> None:
    """Execute YOLO11 fine-tuning with fixed road domain augmentations and early stopping."""
    data_path = Path(data)
    if not data_path.exists():
        print(f"Error: Dataset configuration file '{data}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Inform user about hardware policy
    cuda_available = torch.cuda.is_available()
    if not cuda_available and (device is None or device == "cpu"):
        print(
            "\n[HARDWARE NOTICE] CUDA GPU was not detected. Running on CPU. "
            "For full production training (50 epochs), use Google Colab with GPU acceleration.\n"
        )

    print(f"Loading pretrained model '{model}' for transfer learning...")
    model_obj = YOLO(model)

    print("Beginning fine-tuning with parameters:")
    print(f"  - Dataset:       {data}")
    print(f"  - Model:         {model}")
    print(f"  - Epochs:        {epochs}")
    print(f"  - Image size:    {imgsz}")
    print(f"  - Batch size:    {batch}")
    print(f"  - Early stop:    patience=15")
    print(f"  - Augmentation:  flipud=0.0 (fixed road orientation), fliplr=0.5")
    print(f"  - Plots:         enabled")
    print(f"  - Output run:    {Path(project) / name}")

    # Fine-tune model using transfer learning
    results = model_obj.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        patience=15,       # Early stopping after 15 epochs without validation improvement
        flipud=0.0,        # Disabled: roads maintain fixed vertical perspective
        fliplr=0.5,        # 50% horizontal flip for lateral crack invariance
        plots=True,        # Save training metrics, confusion matrix, and PR curves
        device=device,     # Auto-selects GPU on Colab / remote environment
        workers=workers,
        exist_ok=exist_ok, # Overwrite run folder rather than incrementing names
    )

    save_dir = Path(getattr(results, "save_dir", Path(project) / name))
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = Path(project) / name / "weights" / "best.pt"

    print("\nTraining completed successfully!")
    print(f"Best model weights saved at: {best_pt}")

    # Sync best.pt to local weights/best.pt for seamless default CLI usage
    if best_pt.exists():
        weights_dir = Path("weights")
        weights_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_pt, weights_dir / "best.pt")
        print(f"Synced weights checkpoint to: {weights_dir / 'best.pt'}")

    return results


def main() -> None:
    """CLI entry point for training script."""
    args = parse_args()
    train_yolo(
        data=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        workers=args.workers,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
