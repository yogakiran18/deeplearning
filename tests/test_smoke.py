"""Local Smoke Test for Road Damage Inference.

This test executes local image inference on a sample road image (tests/sample.jpg)
using the lightweight pretrained model 'yolo11n.pt'.

Why this test exists:
- Verifies that the local PyTorch, OpenCV, and Ultralytics environment is correctly configured.
- Validates the end-to-end functionality of src/predict_image.py before training weights are available.
- Confirms that visual annotations (outputs/smoke_test/) and CSV logs (outputs/smoke_test/detections.csv)
  are exported successfully.
"""

import argparse
import subprocess
import sys
import unittest
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for smoke test execution."""
    parser = argparse.ArgumentParser(
        description="Run local inference smoke test using pretrained yolo11n.pt."
    )
    parser.add_argument(
        "--image",
        type=str,
        default="tests/sample.jpg",
        help="Path to sample input image for smoke test (default: 'tests/sample.jpg').",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11n.pt",
        help="Model weights to use for smoke test (default: 'yolo11n.pt').",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/smoke_test",
        help="Output directory for annotated image (default: 'outputs/smoke_test').",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="outputs/smoke_test/detections.csv",
        help="Output path for detection CSV (default: 'outputs/smoke_test/detections.csv').",
    )
    return parser.parse_args()


def run_smoke_test(
    image: str = "tests/sample.jpg",
    weights: str = "yolo11n.pt",
    out_dir: str = "outputs/smoke_test",
    csv_file: str = "outputs/smoke_test/detections.csv",
) -> bool:
    """Execute predict_image.py subprocess and verify output artifacts."""
    image_path = Path(image)
    if not image_path.exists():
        print(f"Error: Sample image '{image}' does not exist.", file=sys.stderr)
        return False

    out_dir_path = Path(out_dir)
    csv_path = Path(csv_file)

    cmd = [
        sys.executable,
        "src/predict_image.py",
        str(image_path),
        "--weights",
        weights,
        "--conf",
        "0.10",
        "--out",
        str(out_dir_path),
        "--csv",
        str(csv_path),
    ]

    print("\n" + "=" * 65)
    print("RUNNING LOCAL INFERENCE SMOKE TEST")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 65)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:\n", result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"[FAIL] Smoke test failed with exit code {result.returncode}!", file=sys.stderr)
        return False

    annotated_image = out_dir_path / image_path.name
    if not annotated_image.exists():
        print(f"[FAIL] Annotated image '{annotated_image}' was not created!", file=sys.stderr)
        return False

    if not csv_path.exists():
        print(f"[FAIL] CSV detection log '{csv_path}' was not created!", file=sys.stderr)
        return False

    print("=" * 65)
    print("[PASS] Smoke test succeeded! Local inference pipeline is functional.")
    print(f"  - Annotated Image: {annotated_image.resolve()}")
    print(f"  - Detection CSV:   {csv_path.resolve()}")
    print("=" * 65 + "\n")
    return True


class TestInferenceSmoke(unittest.TestCase):
    """Unittest test case for CI and test runners."""

    def test_predict_image_smoke(self) -> None:
        success = run_smoke_test(
            image="tests/sample.jpg",
            weights="yolo11n.pt",
            out_dir="outputs/smoke_test",
            csv_file="outputs/smoke_test/detections.csv",
        )
        self.assertTrue(success, "Inference smoke test failed.")

    def test_round_box_pothole_rendering(self) -> None:
        """Verify draw_prominent_box renders round boxes for D40 pothole without errors."""
        import numpy as np
        from src.predict_image import draw_prominent_box

        img = np.zeros((400, 400, 3), dtype=np.uint8)
        draw_prominent_box(img, "D40", 0.94, 50, 50, 200, 150)
        # Verify pixels were drawn
        self.assertGreater(int(img.sum()), 0)

    def test_predict_pothole_sample_with_round_box(self) -> None:
        """Verify pothole sample inference detects D40 and generates annotated output."""
        out_dir = Path("outputs/test_pothole")
        csv_file = out_dir / "detections.csv"
        cmd = [
            sys.executable,
            "src/predict_image.py",
            "samples/pothole.jpg",
            "--out",
            str(out_dir),
            "--csv",
            str(csv_file),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"predict_image failed: {res.stderr}")
        self.assertTrue((out_dir / "pothole.jpg").exists(), "Annotated pothole.jpg was not created")
        self.assertTrue(csv_file.exists(), "CSV detection log was not created")
        csv_content = csv_file.read_text(encoding="utf-8")
        self.assertIn("D40", csv_content, "D40 pothole was not recorded in detection CSV")


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    success = run_smoke_test(
        image=args.image,
        weights=args.weights,
        out_dir=args.out,
        csv_file=args.csv,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
