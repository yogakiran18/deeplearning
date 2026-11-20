# Sample Road Damage Images

This directory provides curated, verified real-world pavement inspection images representing each target distress category:

| File Name | Primary Defect | Standard Code | Bounding Box Annotation | Severity Description |
| :--- | :--- | :---: | :---: | :--- |
| `pothole.jpg` | Pothole | **`D40`** | `pothole.txt` | Deep structural cavity in asphalt |
| `longitudinal_crack.jpg` | Longitudinal Crack | **`D00`** | `longitudinal_crack.txt` | Cracking parallel to vehicle travel direction |
| `transverse_crack.jpg` | Transverse Crack | **`D10`** | `transverse_crack.txt` | Lateral crack across road lanes |
| `alligator_crack.jpg` | Alligator Crack | **`D20`** | `alligator_crack.txt` | Interconnected fatigue mesh cracking |
| `multi_defect_road.jpg` | Multi-Defect Road | **`D00, D10, D20`** | `multi_defect_road.txt` | Multiple combined crack distresses |

---

### Running Single-Image Inference
Test any sample image with YOLO11:

```powershell
# Pothole detection
python src/predict_image.py samples/pothole.jpg --weights weights/best.pt

# Longitudinal crack detection
python src/predict_image.py samples/longitudinal_crack.jpg --weights weights/best.pt

# Transverse crack detection
python src/predict_image.py samples/transverse_crack.jpg --weights weights/best.pt

# Alligator crack detection
python src/predict_image.py samples/alligator_crack.jpg --weights weights/best.pt

# Batch inference across the entire samples folder
python src/predict_image.py samples/ --weights weights/best.pt
```
All predictions will automatically be visualized in `outputs/images/` and recorded in both `outputs/detections.csv` and the SQLite database `outputs/road_damage.db`.
