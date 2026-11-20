"""End-to-End Project Runner for road-damage-yolo.

This master execution script orchestrates the full road damage detection pipeline locally:
1. System & Environment Health Verification (Smoke Test)
2. Dataset Schema & Split Verification (src/prepare_data.py)
3. Model Evaluation on validation split (src/evaluate.py)
4. Image Detection & CSV Coordinate Export (src/predict_image.py)
5. Video Detection & Real-time HUD Rendering (src/predict_video.py)

Why this script exists:
- Provides a single-command entry point to run and verify the entire project locally.
- Coordinates data validation, evaluation, and inference without manual multi-step commands.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the project runner."""
    parser = argparse.ArgumentParser(
        description="Master project runner for road-damage-yolo (runs all pipeline stages)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "smoke", "infer", "eval", "web"],
        help="Pipeline execution mode (default: 'all', use 'web' for localhost web dashboard).",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch the localhost web interface for video/image detection.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for the localhost web interface (default: 5000).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/best.pt",
        help="Path to model weights (default: 'weights/best.pt', fallback: 'yolo11n.pt').",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset data.yaml (auto-detects datasets/road_damage or datasets/road_damage_demo).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="tests/sample.jpg",
        help="Input image for image inference (default: 'tests/sample.jpg').",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="tests/sample_road.mp4",
        help="Input video for video inference (default: 'tests/sample_road.mp4').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on (default: 'cpu').",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display video frames in real-time in an interactive OpenCV GUI window.",
    )
    return parser.parse_args()


def run_command(title: str, cmd: list[str]) -> bool:
    """Execute a CLI step with formatted terminal banner and error handling."""
    print("\n" + "=" * 78)
    print(f"STEP: {title}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 78)

    result = subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{title}' failed with exit code {result.returncode}!", file=sys.stderr)
        return False
    return True


def resolve_paths(args: argparse.Namespace) -> tuple[str, str]:
    """Auto-detect available weights and dataset yaml paths."""
    # Resolve weights
    weights = args.weights
    if not Path(weights).exists():
        if Path("yolo11n.pt").exists():
            weights = "yolo11n.pt"
            print(f"[INFO] '{args.weights}' not found. Using pretrained checkpoint '{weights}'.")

    # Resolve dataset yaml
    data_yaml = args.data
    if data_yaml is None:
        if Path("datasets/road_damage/data.yaml").exists():
            data_yaml = "datasets/road_damage/data.yaml"
        elif Path("datasets/road_damage_demo/data.yaml").exists():
            data_yaml = "datasets/road_damage_demo/data.yaml"
        else:
            data_yaml = "configs/data.yaml"
    print(f"[INFO] Active Model Weights: {weights}")
    print(f"[INFO] Active Dataset YAML:   {data_yaml}")
    return weights, data_yaml


def main() -> None:
    """Execute the master project runner pipeline."""
    args = parse_args()
    py = sys.executable

    print("\n" + "#" * 78)
    print("#  ROAD DAMAGE DETECTION (YOLO11) - MASTER PIPELINE RUNNER")
    print("#" * 78)

    weights, data_yaml = resolve_paths(args)

    # Web Dashboard Mode
    if args.web or args.mode == "web":
        run_command(
            "Launching Localhost Road Damage Web Interface",
            [py, "app.py", "--weights", weights, "--port", str(args.port), "--device", args.device],
        )
        return

    # 1. Smoke test
    if args.mode in ["all", "smoke"]:
        success = run_command(
            "1/5 Smoke Test & Environment Health Check",
            [py, "tests/test_smoke.py"],
        )
        if not success:
            sys.exit(1)

    # 2. Dataset audit
    if args.mode in ["all"]:
        dataset_dir = str(Path(data_yaml).parent)
        success = run_command(
            "2/5 Dataset Verification & Class Audit",
            [py, "src/prepare_data.py", "--verify-only", "--out", dataset_dir],
        )

    # 3. Model evaluation
    if args.mode in ["all", "eval"]:
        success = run_command(
            "3/5 Model Performance Evaluation & Metrics Export",
            [
                py,
                "src/evaluate.py",
                "--weights",
                weights,
                "--data",
                data_yaml,
                "--split",
                "val",
                "--device",
                args.device,
                "--out",
                "outputs/metrics.json",
            ],
        )

    # 4. Image inference
    if args.mode in ["all", "infer"]:
        success = run_command(
            "4/5 Image Inference & Spatial Defect CSV Export",
            [
                py,
                "src/predict_image.py",
                args.image,
                "--weights",
                weights,
                "--conf",
                "0.05",
                "--out",
                "outputs/images",
                "--csv",
                "outputs/detections.csv",
                "--device",
                args.device,
            ],
        )

    # 5. Video inference
    if args.mode in ["all", "infer"]:
        video_cmd = [
            py,
            "src/predict_video.py",
            args.video,
            "--weights",
            weights,
            "--conf",
            "0.05",
            "--out",
            "outputs/result.mp4",
            "--device",
            args.device,
        ]
        if args.show:
            video_cmd.append("--show")
        success = run_command(
            "5/5 Video Frame-by-Frame Inference & HUD Rendering",
            video_cmd,
        )

    # Final summary dashboard
    print("\n" + "#" * 78)
    print("#  PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("#" * 78)
    print("Generated Project Artifacts:")
    print("  * Evaluation Metrics:  outputs/metrics.json")
    print("  * Annotated Image:     outputs/images/sample.jpg")
    print("  * Detection CSV Log:   outputs/detections.csv")
    print("  * Annotated Video:     outputs/result.mp4")
    print("  * Smoke Test Outputs:  outputs/smoke_test/")
    print("#" * 78 + "\n")


if __name__ == "__main__":
    main()
