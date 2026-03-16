# Pipeline Recog YOLO

A complete, interactive YOLOv11 training pipeline for multi-class object detection with manual annotation, dataset generation, training, validation, and ONNX export.

This repository is focused on building detection models from image folders and an annotation workflow. It is not an HTTP API project by default.

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [How the Pipeline Works](#how-the-pipeline-works)
- [Where the Pipeline Gets Its Inputs](#where-the-pipeline-gets-its-inputs)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Generated Outputs](#generated-outputs)
- [Current Endpoints in This Repository](#current-endpoints-in-this-repository)
- [Recommended API Endpoints for Deployment](#recommended-api-endpoints-for-deployment)
- [How to Train for Other Objects](#how-to-train-for-other-objects)
- [Known Constraints and Operational Notes](#known-constraints-and-operational-notes)
- [Troubleshooting](#troubleshooting)

## What This Project Does

The script `pipeline_yolo.py` implements a full training workflow:

1. Loads images from class-specific folders.
2. Opens a Tkinter annotation UI to create one or more bounding boxes per image.
3. Saves annotations into a JSON file.
4. Converts annotations into YOLO label format.
5. Builds train/validation datasets.
6. Trains a YOLOv11 detector.
7. Runs validation metrics.
8. Exports the final model to ONNX.

## How the Pipeline Works

### Stage 1: Class and path configuration

In `pipeline_yolo.py`, classes and folder mapping are defined in code:

- `INPUT_FOLDERS`: source directory per class id
- `CLASSES`: class name per class id
- `PROJECT_NAME`: output directory for training artifacts
- `ANNOTATIONS_FILE`: persistent JSON annotations
- `TRAIN_SPLIT`: train/validation split ratio

### Stage 2: Annotation UI

`MultiClassImageAnnotator` opens a GUI where you can:

- select the class for the next bounding box
- draw multiple boxes per image
- assign different classes in the same image
- save progress continuously to JSON
- resume annotation in later sessions

### Stage 3: Dataset generation

After annotation, the pipeline:

- converts bounding boxes to YOLO normalized format (`class x_center y_center width height`)
- creates directory structure under `PROJECT_NAME`
- copies/normalizes images into `images/train` and `images/val`
- writes labels into `labels/train` and `labels/val`
- optionally adds background images from `input_files/background`
- writes `data.yaml` for Ultralytics training

### Stage 4: Training and validation

The training stage currently initializes from `best.pt` and calls `model.train(...)` with augmentations and fixed hyperparameters.
Epoch count is selected from dataset size.

### Stage 5: Export

The pipeline exports the trained model to ONNX and stores a copy in the project output folder.

## Where the Pipeline Gets Its Inputs

The pipeline relies on these input sources:

1. **Images by class folder**
   - Default mapping in code:
     - `input_files/logo`
     - `input_files/contador`
     - `input_files/caja`

2. **Existing annotations (optional resume)**
   - File: `annotations_multiclass.json`
   - If present, previously annotated images are reused and not re-annotated unless you clear them.

3. **Pretrained YOLO weights for initialization**
   - Current training entrypoint uses `YOLO('best.pt')`.
   - The file `best.pt` must be available at repository root (or the code should be changed to another path).

4. **Optional background images**
   - Folder: `input_files/background`
   - These images are copied without labels as negative examples.

## Project Structure

```text
PipelineRecog/
  pipeline_yolo.py
  requirements.txt
  annotations_multiclass.json
  best.pt
  input_files/
    logo/
    contador/
    caja/
    background/          # optional
```

Training outputs are created under:

```text
mercadolibre_detection_final/
  images/train
  images/val
  labels/train
  labels/val
  data.yaml
  train/
    weights/
      best.pt
      last.pt
  best.onnx
```

## Requirements

From `requirements.txt`:

- Python `>=3.9,<3.12`
- `ultralytics`
- `torch`, `torchvision`
- `opencv-python`
- `Pillow`, `pillow-heif`
- `PyYAML`, `tqdm`
- `onnx`, `onnxruntime`

Additional practical requirement:

- Tkinter GUI support (required for annotation window).

## Installation

```bash
python -m venv .venv
```

Activate:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you need a specific CUDA build of PyTorch, install `torch` and `torchvision` following the official PyTorch CUDA index for your GPU and CUDA version.

## How to Run

### 1) Prepare data folders

Create and populate class folders under `input_files/` according to `INPUT_FOLDERS` in the script.

### 2) Run pipeline

```bash
python pipeline_yolo.py
```

### 3) Annotate images

In the annotation UI:

- choose the class for each next box
- draw all needed boxes on image
- move to next image
- save and exit when done

### 4) Automatic processing

After annotation, dataset creation, training, validation, and ONNX export run automatically.

## Generated Outputs

Main output artifacts:

- `annotations_multiclass.json` (annotation state)
- YOLO dataset (`images/`, `labels/`, `data.yaml`)
- trained model weights (`best.pt`, `last.pt`)
- validation metrics and training plots (Ultralytics output)
- exported ONNX model (`best.onnx`)

## Current Endpoints in This Repository

There are no HTTP endpoints implemented in this repository.

Current execution entrypoint is script-based:

- `python pipeline_yolo.py`

## Recommended API Endpoints for Deployment

If you want to serve this model as an API, a standard FastAPI surface would be:

1. `GET /health`  
   Service health and model load status.

2. `POST /predict`  
   Single image inference (`multipart/form-data`).

3. `POST /predict/base64`  
   Single image inference from base64 payload.

4. `POST /predict/batch`  
   Batch inference for multiple images.

5. `GET /classes`  
   Return class id-to-name mapping used by the model.

6. `PUT /thresholds` (optional)  
   Update confidence/IoU thresholds at runtime.

## How to Train for Other Objects

To adapt this pipeline to a different object set, do the following.

### Step 1: Redefine class mapping

Edit `INPUT_FOLDERS` and `CLASSES` in `pipeline_yolo.py`.

Example:

```python
INPUT_FOLDERS = {
    0: "input_files/person",
    1: "input_files/car",
    2: "input_files/bicycle",
    3: "input_files/dog"
}

CLASSES = {
    0: "person",
    1: "car",
    2: "bicycle",
    3: "dog"
}
```

Rules:

- class ids should be zero-based and contiguous
- each class id must exist in both dictionaries
- each path should contain images for that class

### Step 2: Provide image data

Place images in the new folders under `input_files/`.

### Step 3: Handle annotation state

If you are starting a new labeling project, remove or archive old annotations file:

- `annotations_multiclass.json`

### Step 4: Review pretrained initialization

Training currently starts from `YOLO('best.pt')`.
For a generic start, you may prefer a base model such as `yolo11n.pt` or `yolo11s.pt`.

### Step 5: Run and annotate

Run `python pipeline_yolo.py` and label the new object classes in the GUI.

### Step 6: Validate class outputs

After training, verify:

- generated `data.yaml` names map
- class-wise metrics in validation output
- inference behavior on holdout images

## Known Constraints and Operational Notes

1. The annotation UI is interactive and requires a desktop session; it is not suitable for headless CI environments.
2. `input_files/` must exist and contain images before running.
3. If `input_files/background` does not exist, background stage is skipped.
4. HEIC/HEIF support requires `pillow-heif`.
5. With very small datasets, train and validation may intentionally use the same images.

## Troubleshooting

### No images found

- Confirm `INPUT_FOLDERS` paths exist.
- Confirm file extensions are supported by the script.

### UI does not open

- Ensure Tkinter is installed and available in your Python distribution.
- Run locally with desktop access.

### Training does not start

- Confirm Ultralytics and PyTorch are installed correctly.
- Confirm initial weights file exists (`best.pt` by default).

### ONNX export fails

- Verify `onnx` and `onnxruntime` installation.
- Ensure the trained model exists before export stage.

### Incorrect classes at inference

- Confirm `CLASSES` mapping at training time.
- Verify generated `data.yaml` and label files.
- Re-train if class ids/names changed.
