"""Video Inference and Damage Detection Script for Road Surfaces.

This script processes video footage frame-by-frame using Ultralytics YOLO11 and OpenCV,
annotates detected road damages (D00, D10, D20, D40) with color-coded bounding boxes,
and exports an annotated MP4 video while strictly preserving original FPS and resolution.

Why this script exists:
- Enables continuous inspection of vehicle dashcam footage or drone survey recordings.
- Provides real-time progress visualization with a dynamic tqdm progress bar.
- Maintains a running tally of detections per class both on-screen (HUD) and in terminal.
- Supports an optional --skip N frame rate subsampler to accelerate inference while
  smoothly maintaining original video duration, FPS, and resolution.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO

# Category labels and distinctive BGR colors for clear visual distinction
CLASS_META: Dict[str, Tuple[str, Tuple[int, int, int]]] = {
    "D00": ("Longitudinal Crack", (255, 140, 0)),    # Deep Sky Blue
    "D10": ("Transverse Crack", (50, 205, 50)),      # Lime Green
    "D20": ("Alligator Crack", (0, 165, 255)),       # Orange
    "D40": ("Pothole", (34, 34, 220)),               # Crimson Red
}
DEFAULT_COLOR: Tuple[int, int, int] = (200, 200, 200)


def normalize_class_name(raw_name: str, class_id: int = 0) -> Tuple[str, str]:
    """Map any variant of road damage class name to (standard_code, nice_label)."""
    s = str(raw_name).lower().replace("_", " ").strip()
    if "pothole" in s or s == "d40":
        return "D40", "Pothole"
    elif "alligator" in s or "d20" in s:
        return "D20", "Alligator Crack"
    elif "transverse" in s or "d10" in s:
        return "D10", "Transverse Crack"
    elif "longitudinal" in s or "d00" in s:
        return "D00", "Longitudinal Crack"

    id_map = {
        0: ("D00", "Longitudinal Crack"),
        1: ("D10", "Transverse Crack"),
        2: ("D20", "Alligator Crack"),
        3: ("D40", "Pothole"),
    }
    return id_map.get(class_id, (f"C{class_id}", str(raw_name)))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for video inference."""
    parser = argparse.ArgumentParser(
        description="Run YOLO11 road damage detection on video, annotate frames, and export MP4."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help="Path to input video file (e.g. 'dashcam.mp4').",
    )
    parser.add_argument(
        "--source",
        dest="source_flag",
        type=str,
        default=None,
        help="Optional flag alternative to specify input video file.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/best.pt",
        help="Path to trained YOLO11 weights file (default: 'weights/best.pt').",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for bounding box detection (default: 0.25).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/result.mp4",
        help="Destination path for annotated output MP4 video (default: 'outputs/result.mp4').",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        help="Process every Nth frame with YOLO to speed up inference (default: 1 = every frame).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution in pixels (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on ('0' for GPU, 'cpu' for CPU). Auto-detected by default.",
    )
    parser.add_argument(
        "--no-hud",
        action="store_true",
        help="Disable on-screen HUD overlay showing running class counts.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display video frames in real-time in an interactive OpenCV GUI window (press 'q' to exit).",
    )
    return parser.parse_args()


def draw_detections(
    frame: cv2.typing.MatLike,
    detections: List[Tuple[str, float, int, int, int, int]],
    running_counts: Dict[str, int],
    show_hud: bool = True,
) -> cv2.typing.MatLike:
    """Render bounding boxes, class labels, and optional HUD statistics onto frame."""
    annotated = frame.copy()

    # Draw each detection box and label pill
    h, w = annotated.shape[:2]
    for cls_name, conf, x1, y1, x2, y2 in detections:
        meta = CLASS_META.get(cls_name)
        color = meta[1] if meta else DEFAULT_COLOR
        nice_name = meta[0] if meta else cls_name

        x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
        x2, y2 = min(w - 1, int(round(x2))), min(h - 1, int(round(y2)))

        is_pothole = (cls_name == "D40") or ("pothole" in cls_name.lower())

        if is_pothole:
            # Potholes are surface depressions -> draw prominent round boxes
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            ax = max(6, (x2 - x1) // 2)
            ay = max(6, (y2 - y1) // 2)
            r = min(22, max(8, int(min(x2 - x1, y2 - y1) * 0.35)))

            # Semi-transparent tinted elliptical fill
            overlay = annotated.copy()
            cv2.ellipse(overlay, (cx, cy), (ax, ay), 0, 0, 360, color, -1)
            cv2.addWeighted(overlay, 0.28, annotated, 0.72, 0, annotated)

            # Rounded bounding box outline
            rad = min(r, max(2, (x2 - x1) // 2), max(2, (y2 - y1) // 2))
            for stroke_col, th, off in [((0, 0, 0), 4, 2), (color, 3, 0)]:
                xa, ya = max(0, x1 - off), max(0, y1 - off)
                xb, yb = min(w - 1, x2 + off), min(h - 1, y2 + off)
                cv2.line(annotated, (xa + rad, ya), (xb - rad, ya), stroke_col, th)
                cv2.line(annotated, (xa + rad, yb), (xb - rad, yb), stroke_col, th)
                cv2.line(annotated, (xa, ya + rad), (xa, yb - rad), stroke_col, th)
                cv2.line(annotated, (xb, ya + rad), (xb, yb - rad), stroke_col, th)
                cv2.ellipse(annotated, (xa + rad, ya + rad), (rad, rad), 180, 0, 90, stroke_col, th)
                cv2.ellipse(annotated, (xb - rad, ya + rad), (rad, rad), 270, 0, 90, stroke_col, th)
                cv2.ellipse(annotated, (xb - rad, yb - rad), (rad, rad), 0, 0, 90, stroke_col, th)
                cv2.ellipse(annotated, (xa + rad, yb - rad), (rad, rad), 90, 0, 90, stroke_col, th)

            # Inner targeting ellipse and crosshairs
            cv2.ellipse(annotated, (cx, cy), (ax, ay), 0, 0, 360, (0, 0, 0), 2)
            cv2.ellipse(annotated, (cx, cy), (ax, ay), 0, 0, 360, color, 1)
            cv2.line(annotated, (cx, max(0, y1 - 5)), (cx, min(h - 1, y1 + 5)), (255, 255, 255), 2)
            cv2.line(annotated, (cx, max(0, y2 - 5)), (cx, min(h - 1, y2 + 5)), (255, 255, 255), 2)
            cv2.line(annotated, (max(0, x1 - 5), cy), (min(w - 1, x1 + 5), cy), (255, 255, 255), 2)
            cv2.line(annotated, (max(0, x2 - 5), cy), (min(w - 1, x2 + 5), cy), (255, 255, 255), 2)
        else:
            # Rectangular prominent box for cracks (D00, D10, D20)
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0, annotated)

            cv2.rectangle(annotated, (max(0, x1 - 2), max(0, y1 - 2)), (min(w - 1, x2 + 2), min(h - 1, y2 + 2)), (0, 0, 0), 4)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            c_len = min(22, max(8, int(min(x2 - x1, y2 - y1) * 0.25)))
            for (cx, cy, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                cv2.line(annotated, (cx, cy), (cx + dx * c_len, cy), (255, 255, 255), 4)
                cv2.line(annotated, (cx, cy), (cx, cy + dy * c_len), (255, 255, 255), 4)

        # Bold pill label
        label_text = f" [{cls_name}] {nice_name.upper()} : {int(conf * 100)}% "
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
        pill_y1 = max(0, y1 - text_h - baseline - 8)
        pill_y2 = y1
        pill_x2 = min(w, x1 + text_w + 10)

        cv2.rectangle(annotated, (x1 - 1, pill_y1 - 1), (pill_x2 + 1, pill_y2 + 1), (0, 0, 0), -1)
        cv2.rectangle(annotated, (x1, pill_y1), (pill_x2, pill_y2), color, -1)
        cv2.putText(annotated, label_text, (x1 + 4, pill_y2 - baseline - 3), cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, label_text, (x1 + 4, pill_y2 - baseline - 3), cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # Render semi-transparent top-left HUD banner showing running counts
    if show_hud:
        hud_text = (
            f"Road Damage HUD | D00: {running_counts.get('D00', 0)} | "
            f"D10: {running_counts.get('D10', 0)} | "
            f"D20: {running_counts.get('D20', 0)} | "
            f"D40: {running_counts.get('D40', 0)}"
        )
        (hw, hh), hbase = cv2.getTextSize(
            hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (20 + hw, 20 + hh + hbase), (20, 20, 20), cv2.FILLED)
        # Blend overlay for 65% opacity
        cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)
        cv2.putText(
            annotated,
            hud_text,
            (15, 15 + hh),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated


# Recognized official Ultralytics release weights that can be auto-downloaded
OFFICIAL_MODELS = {
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
}


def process_video(
    source: str,
    weights: str = "weights/best.pt",
    conf: float = 0.25,
    out_path: str = "outputs/result.mp4",
    skip: int = 1,
    imgsz: int = 640,
    device: str | None = None,
    show_hud: bool = True,
    show: bool = False,
) -> None:
    """Read video, run YOLO11 frame detection, write annotated MP4, and print summary."""
    source_path = Path(source)
    if not source_path.exists():
        print(f"Error: Input video '{source}' does not exist.", file=sys.stderr)
        sys.exit(1)

    weights_path = Path(weights)
    if not weights_path.exists() and weights_path.name not in OFFICIAL_MODELS:
        # Check alternative default fallback: runs/train/road_damage_yolo11s/weights/best.pt
        alt_weights = Path("runs/train/road_damage_yolo11s/weights/best.pt")
        if alt_weights.exists():
            print(f"Notice: '{weights}' not found. Using trained run weights at '{alt_weights}'.")
            weights_path = alt_weights
        else:
            print(
                f"Error: Weights file '{weights}' was not found. "
                "Specify valid local weights or a pretrained model (e.g. --weights yolo11n.pt).",
                file=sys.stderr,
            )
            sys.exit(1)

    if skip < 1:
        print("Warning: --skip must be >= 1. Resetting to 1.", file=sys.stderr)
        skip = 1

    # Auto-detect compute hardware
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    print(f"Opening video source '{source}'...")
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        print(f"Error: Failed to open video file '{source}'.", file=sys.stderr)
        sys.exit(1)

    # Extract original video metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps is None:
        fps = 30.0  # Fallback to standard 30 FPS if metadata is missing
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dest = Path(out_path)
    out_dest.parent.mkdir(parents=True, exist_ok=True)

    # Initialize VideoWriter using standard mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_dest), fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"Error: Could not open VideoWriter for '{out_dest}'.", file=sys.stderr)
        cap.release()
        sys.exit(1)

    print(f"Loading YOLO11 model '{weights}' on device '{device}'...")
    model = YOLO(str(weights_path))

    print(f"Video Resolution: {width}x{height} | FPS: {fps:.2f} | Total Frames: {total_frames}")
    print(f"Processing every {skip} frame(s) with confidence threshold={conf}...")

    # Statistics trackers
    running_counts: Dict[str, int] = {"D00": 0, "D10": 0, "D20": 0, "D40": 0}
    total_detections = 0
    cached_detections: List[Tuple[str, float, int, int, int, int]] = []
    frames_written = 0
    inferences_run = 0

    # Progress bar initialized over total frames
    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Processing Video", unit="frame")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Run inference every Nth frame, reusing cached boxes on intermediate skipped frames
        # This keeps the output video at the exact original FPS and duration while accelerating inference
        if frame_idx % skip == 0:
            inferences_run += 1
            results = model.predict(
                source=frame,
                conf=conf,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )

            result = results[0]
            boxes = result.boxes
            cached_detections = []

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    x1, y1, x2, y2 = [int(round(c)) for c in box.xyxy[0].tolist()]
                    cls_name_raw = result.names.get(cls_id, f"class_{cls_id}")
                    cls_code, nice_name = normalize_class_name(cls_name_raw, cls_id)

                    cached_detections.append((cls_code, confidence, x1, y1, x2, y2))
                    running_counts[cls_code] = running_counts.get(cls_code, 0) + 1
                    total_detections += 1

            # Update tqdm progress bar description with dynamic class tallies
            pbar.set_postfix(
                D00=running_counts.get("D00", 0),
                D10=running_counts.get("D10", 0),
                D20=running_counts.get("D20", 0),
                D40=running_counts.get("D40", 0),
            )

        # Draw detections and write frame
        annotated_frame = draw_detections(frame, cached_detections, running_counts, show_hud=show_hud)
        writer.write(annotated_frame)

        # Display interactive GUI window if requested
        if show:
            cv2.imshow("Road Damage Detection (YOLO11) - Real-Time Playback", annotated_frame)
            delay = max(1, int(1000 / fps))
            key = cv2.waitKey(delay) & 0xFF
            if key == ord("q") or key == 27:
                print("\n[INFO] Interactive playback terminated by user.")
                break

        frame_idx += 1
        frames_written += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    # Print comprehensive execution summary
    print("\n" + "=" * 78)
    print("VIDEO INFERENCE COMPLETED SUCCESSFULLY")
    print("=" * 78)
    print(f"  Input Video:          {source_path.resolve()}")
    print(f"  Annotated Video:      {out_dest.resolve()}")
    print(f"  Resolution:           {width}x{height} (original preserved)")
    print(f"  Framerate:            {fps:.2f} FPS (original preserved)")
    print(f"  Total Frames:         {frames_written}")
    print(f"  Inferences Executed:  {inferences_run} (skip={skip})")
    print(f"  Total Detections:     {total_detections}")
    print("-" * 78)
    print("  Per-Class Detection Breakdown:")
    for cls_name in ["D00", "D10", "D20", "D40"]:
        count = running_counts.get(cls_name, 0)
        label = CLASS_META[cls_name][0]
        print(f"    * {cls_name} ({label:<18}): {count}")

    # List any other classes detected outside standard 4
    for cls_name, count in sorted(running_counts.items()):
        if cls_name not in ["D00", "D10", "D20", "D40"]:
            print(f"    * {cls_name:<24}: {count}")

    # Persist inspection in SQLite database
    try:
        from src.database import log_inspection
        deduct = (running_counts.get("D00", 0) * 2) + (running_counts.get("D10", 0) * 2) + (running_counts.get("D20", 0) * 6) + (running_counts.get("D40", 0) * 12)
        pci_score = max(0.0, 100.0 - min(100.0, deduct))
        pci_rating = "Good" if pci_score >= 85 else ("Satisfactory" if pci_score >= 70 else ("Fair" if pci_score >= 55 else "Poor"))

        insp_id = log_inspection({
            "name": source_path.name,
            "media_type": "video",
            "duration_sec": frames_written / max(1.0, fps),
            "total_frames": frames_written,
            "processed_frames": frames_written,
            "fps": fps,
            "pci_score": pci_score,
            "pci_rating": pci_rating,
            "total_defects": total_detections,
            "d00_count": running_counts.get("D00", 0),
            "d10_count": running_counts.get("D10", 0),
            "d20_count": running_counts.get("D20", 0),
            "d40_count": running_counts.get("D40", 0),
            "media_path": str(source_path.resolve()),
            "output_path": str(out_dest.resolve()),
            "model_used": weights,
        })
        print(f"  Database Record:      {insp_id} (outputs/road_damage.db)")
    except Exception as db_err:
        pass

    print("=" * 78 + "\n")


def main() -> None:
    """CLI entry point for video prediction."""
    args = parse_args()
    source = args.source_flag or args.source
    if not source:
        print("Error: Input video path is required. Provide as positional argument or via --source.", file=sys.stderr)
        sys.exit(1)

    process_video(
        source=source,
        weights=args.weights,
        conf=args.conf,
        out_path=args.out,
        skip=args.skip,
        imgsz=args.imgsz,
        device=args.device,
        show_hud=not args.no_hud,
        show=args.show,
    )


if __name__ == "__main__":
    main()
