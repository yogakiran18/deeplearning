"""Model Evaluation Script for Road Damage Detection.

This script evaluates a trained Ultralytics YOLO11 model on a dataset split (default: 'test').
It computes key detection metrics (Precision, Recall, mAP@0.5, mAP@0.5:0.95), prints a formatted
per-class performance table, and saves the metrics to a structured JSON file (outputs/metrics.json)
for project reports and technical documentation.

Why this script exists:
- Provides an automated, reproducible benchmark of trained YOLO11 models on held-out test data.
- Generates publication-ready metrics for road damage classes (D00, D10, D20, D40).
- Exports machine-readable JSON metrics for reporting, charting, and error analysis.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict
import torch
import yaml
from ultralytics import YOLO

# Standard class descriptions for clarity in reporting
CLASS_DESCRIPTIONS: Dict[str, str] = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
}

# Recognized official Ultralytics release weights that can be auto-downloaded
OFFICIAL_MODELS = {
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for model evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained YOLO11 weights on the test split and export metrics to JSON."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/best.pt",
        help="Path to trained YOLO11 model weights file (default: 'weights/best.pt').",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/road_damage/data.yaml",
        help="Path to data.yaml dataset specification (default: 'datasets/road_damage/data.yaml').",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "valid"],
        help="Dataset split to evaluate against (default: 'test').",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution in pixels (default: 640).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for evaluation dataloader (default: 16).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device ('0' for GPU, 'cpu' for CPU). Auto-detects by default.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/metrics.json",
        help="Destination path for output metrics JSON file (default: 'outputs/metrics.json').",
    )
    return parser.parse_args()


def evaluate(
    weights: str,
    data: str,
    split: str = "test",
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    out_json: str = "outputs/metrics.json",
) -> Dict[str, Any]:
    """Run validation on the specified split and export metrics."""
    weights_path = Path(weights)
    data_path = Path(data)

    # Normalize split name ('valid' -> 'val')
    if split == "valid":
        split = "val"

    # Validate weights path exists or is a standard Ultralytics checkpoint name
    if not weights_path.exists() and weights_path.name not in OFFICIAL_MODELS:
        # Check alternative default fallback: runs/train/road_damage_yolo11s/weights/best.pt
        alt_weights = Path("runs/train/road_damage_yolo11s/weights/best.pt")
        if alt_weights.exists():
            print(f"Notice: '{weights}' not found. Using trained run weights at '{alt_weights}'.")
            weights_path = alt_weights
        else:
            print(
                f"Error: Weights file '{weights}' was not found. "
                "Ensure training has finished on Google Colab and weights are downloaded.",
                file=sys.stderr,
            )
            sys.exit(1)

    if not data_path.exists():
        print(f"Error: Dataset configuration file '{data}' was not found.", file=sys.stderr)
        sys.exit(1)

    # Inspect data.yaml for split availability
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f) or {}
        if split == "test" and "test" not in data_cfg:
            print("[INFO] 'test' key not found in data.yaml. Evaluating on 'val' split instead.")
            split = "val"
    except Exception as e:
        print(f"Notice: Could not parse data.yaml ({e}), proceeding with requested split '{split}'.")

    # Auto-detect device if unspecified
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    print(f"Loading weights from '{weights_path}' on device '{device}'...")
    model = YOLO(str(weights_path))

    print(f"Evaluating model on split '{split}' using '{data}' (imgsz={imgsz}, batch={batch})...")
    # Run evaluation using Ultralytics validation engine
    val_results = model.val(
        data=str(data_path),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        plots=True,
    )

    box = val_results.box
    names = val_results.names

    # Extract overall scalar metrics
    overall_precision = float(box.mp)
    overall_recall = float(box.mr)
    overall_map50 = float(box.map50)
    overall_map50_95 = float(box.map)

    # Print summary of overall metrics
    print("\n" + "=" * 78)
    print(f"OVERALL EVALUATION METRICS (Split: {split})")
    print("=" * 78)
    print(f"  Precision:       {overall_precision:.4f}")
    print(f"  Recall:          {overall_recall:.4f}")
    print(f"  mAP@0.5:         {overall_map50:.4f}")
    print(f"  mAP@0.5:0.95:    {overall_map50_95:.4f}")
    print("-" * 78)

    # Extract per-class performance metrics
    per_class_metrics: Dict[str, Dict[str, Any]] = {}
    print(f"{'Class':<22} {'Description':<20} {'Precision':<10} {'Recall':<10} {'AP@0.5':<10} {'AP@0.5:0.95':<12}")
    print("-" * 78)

    # ap_class_index maps indices of classes that appeared in evaluation
    for idx, class_id in enumerate(box.ap_class_index):
        class_name = str(names.get(class_id, f"class_{class_id}"))
        p, r, ap50, ap = box.class_result(idx)
        desc = CLASS_DESCRIPTIONS.get(class_name, "Road defect")

        per_class_metrics[class_name] = {
            "class_id": int(class_id),
            "description": desc,
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "ap50": round(float(ap50), 4),
            "ap50_95": round(float(ap), 4),
        }

        print(
            f"{class_name:<22} {desc:<20} {float(p):<10.4f} {float(r):<10.4f} {float(ap50):<10.4f} {float(ap):<12.4f}"
        )

    print("-" * 78)
    print(
        f"{'all (mean)':<22} {'All classes':<20} {overall_precision:<10.4f} {overall_recall:<10.4f} {overall_map50:<10.4f} {overall_map50_95:<12.4f}"
    )
    print("=" * 78 + "\n")

    # Assemble structured dictionary for JSON export
    metrics_payload = {
        "model_weights": str(weights_path),
        "dataset_yaml": str(data_path),
        "split": split,
        "imgsz": imgsz,
        "overall": {
            "precision": round(overall_precision, 4),
            "recall": round(overall_recall, 4),
            "map50": round(overall_map50, 4),
            "map50_95": round(overall_map50_95, 4),
        },
        "per_class": per_class_metrics,
    }

    # Save metrics to target JSON file
    out_file = Path(out_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"Metrics successfully exported to: {out_file.resolve()}\n")
    return metrics_payload


def main() -> None:
    """CLI entry point for evaluate script."""
    args = parse_args()
    evaluate(
        weights=args.weights,
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        out_json=args.out,
    )


if __name__ == "__main__":
    main()
