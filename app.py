"""Road Damage Detection System (YOLO11) - Localhost Web Application Launcher.

Run this script to launch the local web server:
    python app.py

Then open your browser at:
    http://127.0.0.1:5000

Features:
- Video Upload & Real-Time Progress: Upload road survey videos or use the bundled demo video.
- Live Detection Preview: Real-time HUD and bounding boxes for D00, D10, D20, and D40.
- Browser Playback: Automatic H.264 MP4 transcoding for native HTML5 video playback.
- Image Inspection: Single/batch photo analysis with side-by-side comparisons.
- Live Webcam: Real-time road damage detection feed from your webcam.
- PCI Road Health Assessment: Pavement Condition Index score and CSV export.
"""

import argparse
import sys
import webbrowser
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.web.app import create_app


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch Road Damage Detection Localhost Web Interface."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind server to (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port number to listen on (default: 5000).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/best.pt",
        help="Path to YOLO11 weights (default: weights/best.pt).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('0' for GPU, 'cpu' for CPU). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the default web browser on startup.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Flask server in debug mode.",
    )
    return parser.parse_args()


def main():
    """Main launcher."""
    args = parse_args()

    url = f"http://{args.host}:{args.port}"
    print("\n" + "=" * 72)
    print("   🛣️  ROAD DAMAGE DETECTION (YOLO11) - LOCALHOST WEB DASHBOARD")
    print("=" * 72)
    print(f" * Server URL      : {url}")
    print(f" * Model Weights   : {args.weights}")
    print(" * Supported Modes : Video Upload, Image Inspection, Live Webcam")
    print("=" * 72 + "\n")

    if not args.no_browser and not args.debug:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    app = create_app(weights_path=args.weights, device=args.device)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
