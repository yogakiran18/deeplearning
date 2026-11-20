"""Unit tests for dataset preparation, Kaggle handling, and VOC XML conversion."""

import tempfile
from pathlib import Path
import yaml
import pytest

from src.prepare_data import (
    convert_voc_xml_to_yolo,
    convert_all_voc_in_directory,
    post_process_dataset,
    verify_dataset,
    EXPECTED_CLASSES,
)


def test_convert_voc_xml_to_yolo():
    """Verify that Pascal VOC XML bounding boxes are correctly converted to YOLO format."""
    xml_content = """<annotation>
    <size>
        <width>1000</width>
        <height>500</height>
    </size>
    <object>
        <name>D00</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>100</ymin>
            <xmax>300</xmax>
            <ymax>300</ymax>
        </bndbox>
    </object>
    <object>
        <name>pothole</name>
        <bndbox>
            <xmin>500</xmin>
            <ymin>200</ymin>
            <xmax>700</xmax>
            <ymax>400</ymax>
        </bndbox>
    </object>
</annotation>"""

    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as tmp:
        tmp.write(xml_content)
        tmp_path = Path(tmp.name)

    try:
        yolo_lines = convert_voc_xml_to_yolo(tmp_path)
        assert len(yolo_lines) == 2

        # D00 -> class 0: xc=(100+300)/2/1000 = 0.2, yc=(100+300)/2/500 = 0.4, w=200/1000=0.2, h=200/500=0.4
        assert yolo_lines[0].startswith("0 0.200000 0.400000 0.200000 0.400000")

        # pothole -> class 3: xc=(500+700)/2/1000 = 0.6, yc=(200+400)/2/500 = 0.6, w=200/1000=0.2, h=200/500=0.4
        assert yolo_lines[1].startswith("3 0.600000 0.600000 0.200000 0.400000")
    finally:
        tmp_path.unlink(missing_ok=True)


def test_post_process_dataset_generates_data_yaml():
    """Verify that post_process_dataset creates a valid data.yaml if missing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        train_img = root / "train" / "images"
        val_img = root / "valid" / "images"
        train_lbl = root / "train" / "labels"
        val_lbl = root / "valid" / "labels"

        train_img.mkdir(parents=True)
        val_img.mkdir(parents=True)
        train_lbl.mkdir(parents=True)
        val_lbl.mkdir(parents=True)

        # Create dummy image and label
        (train_img / "img1.jpg").write_bytes(b"dummy")
        (train_lbl / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (val_img / "img2.jpg").write_bytes(b"dummy")
        (val_lbl / "img2.txt").write_text("3 0.5 0.5 0.2 0.2\n")

        post_process_dataset(root)

        yaml_path = root / "data.yaml"
        assert yaml_path.exists()

        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert cfg["names"] == {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
        assert verify_dataset(root) is True

