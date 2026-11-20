"""Road Damage Dataset Preparation and Verification Script.

This script automates downloading, extracting, organizing, and verifying road damage
detection datasets from both Kaggle and Roboflow Universe for YOLO11 training.

Target Classes:
- 0: D00 (Longitudinal crack)
- 1: D10 (Transverse crack)
- 2: D20 (Alligator crack)
- 3: D40 (Pothole)

Why this script exists:
- Provides a reproducible CLI command to fetch datasets from Kaggle or Roboflow without manual downloads.
- Handles Kaggle authentication via environment variables (KAGGLE_USERNAME, KAGGLE_KEY) or kaggle.json.
- Automatically handles nested dataset archives, VOC XML to YOLO txt annotation conversion, and data.yaml generation.
- Validates split presence (train, valid/val, test) and 1:1 image-to-label alignment.
- Audits data.yaml to ensure compatibility with target classes (D00, D10, D20, D40).
"""

import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Supported image file formats
IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Expected road damage target classes
EXPECTED_CLASSES: List[str] = ["D00", "D10", "D20", "D40"]

# Normalization mapping for RDD classes (case-insensitive & aliases)
CLASS_MAP: Dict[str, int] = {
    "d00": 0,
    "longitudinal": 0,
    "longitudinal crack": 0,
    "d10": 1,
    "transverse": 1,
    "transverse crack": 1,
    "d20": 2,
    "alligator": 2,
    "alligator crack": 2,
    "d40": 3,
    "pothole": 3,
    "potholes": 3,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset download, preparation, and verification."""
    parser = argparse.ArgumentParser(
        description="Download, organize, and verify a road damage dataset from Kaggle or Roboflow for YOLO11."
    )

    # Kaggle arguments
    kaggle_group = parser.add_argument_group("Kaggle Dataset Options")
    kaggle_group.add_argument(
        "--kaggle",
        "--kaggle_dataset",
        type=str,
        dest="kaggle_dataset",
        default=None,
        help="Kaggle dataset slug (e.g. 'aliabdelmenam/rdd-2022' or 'chriskue/road-damage-dataset').",
    )
    kaggle_group.add_argument(
        "--kaggle_token",
        type=str,
        default=os.getenv("KAGGLE_API_TOKEN"),
        help="Kaggle API Access Token (e.g. 'KGAT_...'). Falls back to KAGGLE_API_TOKEN environment variable.",
    )
    kaggle_group.add_argument(
        "--kaggle_username",
        type=str,
        default=os.getenv("KAGGLE_USERNAME"),
        help="Kaggle API username. Falls back to KAGGLE_USERNAME environment variable.",
    )
    kaggle_group.add_argument(
        "--kaggle_key",
        type=str,
        default=os.getenv("KAGGLE_KEY"),
        help="Kaggle API key. Falls back to KAGGLE_KEY environment variable.",
    )

    # Roboflow arguments
    rf_group = parser.add_argument_group("Roboflow Dataset Options")
    rf_group.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("ROBOFLOW_API_KEY"),
        help="Roboflow Private API key. Falls back to ROBOFLOW_API_KEY environment variable.",
    )
    rf_group.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Roboflow workspace URL slug (e.g., 'workspace-name').",
    )
    rf_group.add_argument(
        "--project",
        type=str,
        default=None,
        help="Roboflow project URL slug (e.g., 'road-damage-detection').",
    )
    rf_group.add_argument(
        "--version",
        type=int,
        default=None,
        help="Roboflow dataset version number (integer).",
    )
    rf_group.add_argument(
        "--model_format",
        type=str,
        default="yolov11",
        help="Roboflow export format (default: 'yolov11', fallback: 'yolov8').",
    )

    # General options
    parser.add_argument(
        "--out",
        type=str,
        default="datasets/road_damage",
        help="Destination directory for downloaded dataset (default: 'datasets/road_damage').",
    )
    parser.add_argument(
        "--auto_convert_voc",
        action="store_true",
        default=True,
        help="Automatically convert Pascal VOC XML annotations to YOLO txt format if XMLs are detected.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip download and perform verification on existing dataset in --out directory.",
    )

    return parser.parse_args()


def download_from_kaggle(
    dataset_slug: str,
    out_dir: Path,
    username: Optional[str] = None,
    key: Optional[str] = None,
    token: Optional[str] = None,
) -> Path:
    """Download and unpack a road damage dataset from Kaggle."""
    # Apply explicit credentials if provided
    if token:
        os.environ["KAGGLE_API_TOKEN"] = token
    if username:
        os.environ["KAGGLE_USERNAME"] = username
    if key:
        os.environ["KAGGLE_KEY"] = key

    # Check for credentials before attempting download
    has_token = bool(
        os.getenv("KAGGLE_API_TOKEN")
        or (Path.home() / ".kaggle" / "access_token").exists()
        or (Path.home() / ".kaggle" / "access_token.txt").exists()
    )
    has_legacy = bool(
        (Path.home() / ".kaggle" / "kaggle.json").exists()
        or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    )

    if not (has_token or has_legacy):
        print(
            "\n" + "!" * 70 + "\n"
            "Error: Kaggle credentials not found!\n"
            "To download datasets from Kaggle:\n"
            "1. Visit https://www.kaggle.com -> Click your Profile Icon -> Settings.\n"
            "2. Scroll down to 'API' section -> Click 'Create New Token'.\n"
            "   (Or copy your Kaggle Access Token KGAT_...)\n"
            "3. Set your credentials:\n"
            "   - In Google Colab Secrets (Key icon in left sidebar):\n"
            "     Add 'KAGGLE_API_TOKEN' (or 'KAGGLE_USERNAME' and 'KAGGLE_KEY')\n"
            "   - In .env file:\n"
            "     KAGGLE_API_TOKEN=KGAT_...\n"
            "   - In PowerShell:\n"
            "     $env:KAGGLE_API_TOKEN=\"KGAT_...\"\n"
            "   - Or place access_token / kaggle.json in ~/.kaggle/\n"
            + "!" * 70 + "\n",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except Exception as exc:
        print(f"Error authenticating with Kaggle API: {exc}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Kaggle] Downloading '{dataset_slug}' into '{out_dir}'...")

    try:
        api.dataset_download_files(dataset_slug, path=str(out_dir), unzip=True, quiet=False)
        print("[Kaggle] Download and extraction complete.")
    except Exception as exc:
        print(f"Error downloading Kaggle dataset '{dataset_slug}': {exc}", file=sys.stderr)
        sys.exit(1)

    # Post-process structure (flatten nesting, convert XMLs, verify data.yaml)
    post_process_dataset(out_dir)
    return out_dir


def download_from_roboflow(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    out_dir: Path,
    model_format: str = "yolov11",
) -> Path:
    """Download the specified dataset version from Roboflow in YOLO format."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print(
            "Error: 'roboflow' is not installed in the current environment. "
            "Please run: pip install roboflow",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Connecting to Roboflow workspace '{workspace}', project '{project}', version {version}...")
    rf = Roboflow(api_key=api_key)
    rf_workspace = rf.workspace(workspace)
    rf_project = rf_workspace.project(project)
    rf_version = rf_project.version(version)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset format '{model_format}' to '{out_dir}'...")
    try:
        dataset = rf_version.download(
            model_format=model_format,
            location=str(out_dir),
            overwrite=True,
        )
        return Path(dataset.location)
    except Exception as exc:
        if model_format == "yolov11":
            print(f"Notice: yolov11 export failed ({exc}). Retrying with fallback format 'yolov8'...")
            dataset = rf_version.download(
                model_format="yolov8",
                location=str(out_dir),
                overwrite=True,
            )
            return Path(dataset.location)
        raise exc


def convert_voc_xml_to_yolo(xml_file: Path) -> List[str]:
    """Parse a Pascal VOC XML annotation file and return lines in YOLO format."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except Exception:
        return []

    size_elem = root.find("size")
    if size_elem is None:
        return []

    try:
        width = float(size_elem.findtext("width", "0"))
        height = float(size_elem.findtext("height", "0"))
    except ValueError:
        return []

    if width <= 0 or height <= 0:
        return []

    yolo_lines = []
    for obj in root.findall("object"):
        raw_name = obj.findtext("name", "").strip().lower()
        if raw_name not in CLASS_MAP:
            continue

        cls_id = CLASS_MAP[raw_name]
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        try:
            xmin = float(bndbox.findtext("xmin", "0"))
            ymin = float(bndbox.findtext("ymin", "0"))
            xmax = float(bndbox.findtext("xmax", "0"))
            ymax = float(bndbox.findtext("ymax", "0"))
        except ValueError:
            continue

        if xmax <= xmin or ymax <= ymin:
            continue

        # Compute normalized YOLO coordinates (clamped between 0 and 1)
        x_center = max(0.0, min(1.0, ((xmin + xmax) / 2.0) / width))
        y_center = max(0.0, min(1.0, ((ymin + ymax) / 2.0) / height))
        box_w = max(0.0, min(1.0, (xmax - xmin) / width))
        box_h = max(0.0, min(1.0, (ymax - ymin) / height))

        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

    return yolo_lines


def convert_all_voc_in_directory(dataset_dir: Path) -> int:
    """Find all Pascal VOC XML files in the dataset directory and convert them to YOLO format."""
    xml_files = list(dataset_dir.rglob("*.xml"))
    if not xml_files:
        return 0

    print(f"[Dataset Structuring] Found {len(xml_files)} Pascal VOC XML annotation files. Converting to YOLO format...")
    converted_count = 0

    for xml_file in xml_files:
        yolo_lines = convert_voc_xml_to_yolo(xml_file)
        # Determine appropriate target label file
        parent_dir = xml_file.parent
        # If xml is in an 'annotations' or 'xmls' dir, place txt in a sibling 'labels' dir
        if parent_dir.name in {"xmls", "annotations", "Annotations"}:
            split_dir = parent_dir.parent
            labels_dir = split_dir / "labels"
            labels_dir.mkdir(parents=True, exist_ok=True)
            txt_path = labels_dir / f"{xml_file.stem}.txt"
        else:
            txt_path = parent_dir / f"{xml_file.stem}.txt"

        with open(txt_path, "w", encoding="utf-8") as f:
            for line in yolo_lines:
                f.write(line + "\n")
        converted_count += 1

    print(f"[Dataset Structuring] Converted {converted_count} XML files to YOLO labels.")
    return converted_count


def post_process_dataset(dataset_dir: Path) -> None:
    """Organize extracted Kaggle dataset into a standard YOLO structure with data.yaml."""
    print("\n[Dataset Structuring] Auditing and organizing dataset structure...")

    # 1. Flatten single nested root if archive unzipped into an inner subfolder
    subdirs = [p for p in dataset_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    files = [p for p in dataset_dir.iterdir() if p.is_file() and not p.name.startswith(".")]
    if len(subdirs) == 1 and not files:
        nested_dir = subdirs[0]
        # Check if inner dir contains train/images/etc.
        inner_contents = list(nested_dir.iterdir())
        if any(p.is_dir() for p in inner_contents) or any(p.name.endswith(".yaml") for p in inner_contents):
            print(f"[Dataset Structuring] Flattening nested directory '{nested_dir.name}'...")
            for item in inner_contents:
                shutil.move(str(item), str(dataset_dir / item.name))
            nested_dir.rmdir()

    # 2. Check and convert any VOC XML files
    convert_all_voc_in_directory(dataset_dir)

    # 3. Locate or generate data.yaml
    yaml_path = find_data_yaml(dataset_dir)
    if not yaml_path:
        # Detect split directories
        train_path = None
        val_path = None
        test_path = None

        for cand in ["train", "images/train", "train/images"]:
            if (dataset_dir / cand).is_dir():
                train_path = cand
                break

        for cand in ["valid", "val", "images/valid", "images/val", "valid/images", "val/images"]:
            if (dataset_dir / cand).is_dir():
                val_path = cand
                break

        for cand in ["test", "images/test", "test/images"]:
            if (dataset_dir / cand).is_dir():
                test_path = cand
                break

        # Fallback if flat image folder
        if not train_path:
            images_dir = dataset_dir / "images"
            if images_dir.is_dir():
                train_path = "images"
                val_path = "images"

        new_yaml = dataset_dir / "data.yaml"
        yaml_content = {
            "path": str(dataset_dir.resolve()),
            "train": train_path or "train",
            "val": val_path or train_path or "val",
            "names": {0: "D00", 1: "D10", 2: "D20", 3: "D40"},
        }
        if test_path:
            yaml_content["test"] = test_path

        with open(new_yaml, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        print(f"[Dataset Structuring] Generated standard data.yaml at '{new_yaml.name}'.")
    else:
        print(f"[Dataset Structuring] Existing configuration found at '{yaml_path.name}'.")


def find_data_yaml(dataset_dir: Path) -> Optional[Path]:
    """Locate data.yaml or equivalent configuration in the dataset root."""
    standard_yaml = dataset_dir / "data.yaml"
    if standard_yaml.is_file():
        return standard_yaml

    candidates = list(dataset_dir.glob("*.yaml")) + list(dataset_dir.glob("*.yml"))
    return candidates[0] if candidates else None


def load_classes(yaml_path: Path) -> List[str]:
    """Parse class names from dataset YAML file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    raw_names = data_cfg.get("names", [])
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    elif isinstance(raw_names, dict):
        return [str(raw_names[k]) for k in sorted(raw_names.keys())]
    return []


def count_files_in_split(split_dir: Path) -> Tuple[int, int]:
    """Count image files and text label files in a split directory."""
    images_dir = split_dir / "images" if (split_dir / "images").is_dir() else split_dir
    labels_dir = split_dir / "labels" if (split_dir / "labels").is_dir() else split_dir

    images = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    labels = [p for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
    return len(images), len(labels)


def verify_dataset(dataset_dir: Path) -> bool:
    """Verify dataset directory structure, split counts, and class definitions."""
    print("\n" + "=" * 60)
    print(f"VERIFYING DATASET INTEGRITY AT: {dataset_dir.resolve()}")
    print("=" * 60)

    has_warnings = False

    # 1. Inspect data.yaml and classes
    yaml_path = find_data_yaml(dataset_dir)
    if not yaml_path:
        print("[WARNING] data.yaml not found in dataset root directory.")
        has_warnings = True
    else:
        classes = load_classes(yaml_path)
        print(f"\n[INFO] data.yaml located at: {yaml_path.name}")
        print(f"[INFO] Detected class count: {len(classes)}")
        print(f"[INFO] Detected class names: {classes}")

        if classes != EXPECTED_CLASSES:
            has_warnings = True
            print("\n[WARNING] Dataset classes differ from expected classes:")
            print(f"  Expected: {EXPECTED_CLASSES}")
            print(f"  Found:    {classes}")
            missing = set(EXPECTED_CLASSES) - set(classes)
            extra = set(classes) - set(EXPECTED_CLASSES)
            if missing:
                print(f"  Missing required classes: {sorted(missing)}")
            if extra:
                print(f"  Extra/non-standard classes: {sorted(extra)}")
        else:
            print("[OK] Class names match target specification (D00, D10, D20, D40).")

    # 2. Verify train / valid / test splits
    print("\n--- Split Directory Verification ---")
    splits = ["train", "valid", "test"]
    split_counts: Dict[str, Tuple[int, int]] = {}

    for split in splits:
        split_path = dataset_dir / split
        if split == "valid" and not split_path.is_dir() and (dataset_dir / "val").is_dir():
            split_path = dataset_dir / "val"

        if not split_path.is_dir():
            if split == "test":
                print("[INFO] Optional split folder 'test' is not present.")
            else:
                print(f"[WARNING] Required split folder '{split}' is missing (checked '{split_path.name}').")
                has_warnings = True
            continue

        num_images, num_labels = count_files_in_split(split_path)
        split_counts[split] = (num_images, num_labels)

        if num_images == 0:
            print(f"[WARNING] Split '{split}' contains 0 images.")
            has_warnings = True
        elif num_images != num_labels:
            print(
                f"[WARNING] Split '{split}' count mismatch: {num_images} images vs {num_labels} labels."
            )
            has_warnings = True
        else:
            print(f"[OK] Split '{split}': {num_images} images, {num_labels} labels (1:1 match).")

    # Summary table
    print("\n--- Image Distribution Summary ---")
    for split, (imgs, lbls) in split_counts.items():
        print(f"  - Split '{split}': {imgs} images ({lbls} labels)")

    print("=" * 60)
    if has_warnings:
        print("Dataset verification completed with warnings. Review details above.")
    else:
        print("Dataset verification PASSED with no warnings.")
    print("=" * 60 + "\n")
    return not has_warnings


def main() -> None:
    """Execute main CLI workflow for dataset preparation and verification."""
    args = parse_args()
    out_dir = Path(args.out)

    if not args.verify_only:
        # Check if Kaggle download was requested
        if args.kaggle_dataset:
            download_from_kaggle(
                dataset_slug=args.kaggle_dataset,
                out_dir=out_dir,
                username=args.kaggle_username,
                key=args.kaggle_key,
                token=args.kaggle_token,
            )
        # Check if Roboflow download was requested
        elif args.workspace and args.project and args.version is not None:
            api_key = args.api_key or os.getenv("ROBOFLOW_API_KEY")
            if not api_key:
                print(
                    "Error: Roboflow API key is required. Provide it via --api_key or "
                    "set the ROBOFLOW_API_KEY environment variable.",
                    file=sys.stderr,
                )
                sys.exit(1)

            download_from_roboflow(
                api_key=api_key,
                workspace=args.workspace,
                project=args.project,
                version=args.version,
                out_dir=out_dir,
                model_format=args.model_format,
            )
        else:
            print(
                "\n" + "=" * 65 + "\n"
                "Error: Please specify a dataset source to download, or use --verify-only.\n\n"
                "Option 1: Download from Kaggle (Recommended)\n"
                "  python src/prepare_data.py --kaggle aliabdelmenam/rdd-2022 --out datasets/road_damage\n\n"
                "Option 2: Download from Roboflow Universe\n"
                "  python src/prepare_data.py --workspace <ws> --project <proj> --version <v> --out datasets/road_damage\n\n"
                "Option 3: Verify existing dataset folder\n"
                "  python src/prepare_data.py --verify-only --out datasets/road_damage_demo\n"
                + "=" * 65 + "\n",
                file=sys.stderr,
            )
            sys.exit(1)

    if not out_dir.exists():
        print(f"Error: Dataset directory '{out_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    verify_dataset(out_dir)


if __name__ == "__main__":
    main()
