# Project Rules: road-damage-yolo

## Core Objectives & Scope
- **Task**: Road damage detection using Ultralytics YOLO11.
- **Classes**:
  - `0`: `D00` (longitudinal crack)
  - `1`: `D10` (transverse crack)
  - `2`: `D20` (alligator crack)
  - `3`: `D40` (pothole)

## Execution Environment & Hardware
- **Training**: Strictly on **Google Colab (GPU)**. Never run training locally.
- **Local Machine**: Dedicated for code editing, data exploration/verification, and local inference testing only.

## Code Standards
- **Python**: Version 3.10+.
- **Structure**: Scripts reside in `src/`.
- **CLI & Documentation**: Every script must include:
  - Clear module docstring explaining what it does and why.
  - Command-line interface implemented via `argparse`.
- **Security**: Never hardcode API keys or credentials. Use environment variables (e.g. `os.environ`, `.env`) or Colab Secrets.
- **Style**: Simple, clean, readable code with short comments clarifying non-obvious design choices.

## Workflow & Communication
- After completing each task:
  - Summarize what changed.
  - Provide exact instructions on how to run or use the updated scripts/notebooks.
