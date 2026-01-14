## YOLO Buoy Validation (PyQt6)

This repo contains:
- `my_model.pt`: a YOLO model trained to detect buoys
- `images/`: a folder of test images
- `prepare_images.py`: renames images to `1.jpg, 2.jpg, ...` for consistent indexing
- `label_app.py`: a GUI to visualize detections and record ground-truth labels into `labels.csv`

## Setup (Windows PowerShell)

From the repo root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Step 1: Rename images to 1.jpg, 2.jpg, ...

Dry-run first (recommended):

```powershell
python .\prepare_images.py --dry-run
```

Then rename:

```powershell
python .\prepare_images.py
```

## Step 2: Run the labeling app

```powershell
python .\label_app.py
```

## Output CSV

The app creates/updates `labels.csv` in the repo root with columns:
- `image_number`: integer derived from filename (e.g. `12` for `12.jpg`)
- `buoy_actual`: your ground truth (usually `0` or `1`, can be decimal for edge cases)
- `buoy_detected`: detection score in `[0,1]` (max confidence across detections; `0` if none)
- `certainty_avg`: average confidence across detections (`0` if none)

## Controls (default)

- `A`: mark **Actual buoy present = 1**
- `S`: mark **Actual buoy present = 0**
- `D`: apply **edge-case decimal** from the input box as `buoy_actual`
- `Left/Right`: previous/next image
- `Ctrl+S`: save current label (also auto-saves on button presses)


