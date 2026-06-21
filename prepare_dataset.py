# Vehicle Detection with YOLOv8

This project fine-tunes YOLOv8 on a vehicle dataset with exactly five classes:

| Class ID | Class |
|---:|---|
| 0 | Car |
| 1 | Bicycle |
| 2 | Bus |
| 3 | Truck |
| 4 | Motorcycle |

The project trains and compares three YOLOv8 scales:

- YOLOv8s: small, fastest, lowest resource use
- YOLOv8m: medium, balanced speed and accuracy
- YOLOv8l: large, slower, usually most accurate

## Phase 1: Dataset Preparation

### Labeling

Use one of these annotation tools:

- Roboflow: easiest for upload, annotation, split, and YOLOv8 export
- CVAT: strong professional annotation tool
- LabelImg: simple local labeling tool

For each object, draw a tight bounding box around the visible vehicle. Export labels in YOLO format.

Each label file must have the same filename stem as the image:

```text
images/train/img_001.jpg
labels/train/img_001.txt
```

Each row in a YOLO label file must be:

```text
class_id x_center y_center width height
```

The four coordinate values are normalized to the image width and height, so they must be between `0` and `1`.

### Recommended Split

- Train: 70%
- Validation: 20%
- Test: 10%

The training set updates model weights. The validation set is used during training for model selection. The test set is used only for final comparison.

### Required Folder Structure

```text
vehicle_dataset/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### `data.yaml`

```yaml
path: /content/vehicle_dataset

train: images/train
val: images/val
test: images/test

names:
  0: Car
  1: Bicycle
  2: Bus
  3: Truck
  4: Motorcycle
```

For your Windows path, use forward slashes:

```yaml
path: C:/Users/savan/Desktop/CS/Deep learning/Exercises/دیتاست تمرین 5/vehicle-images
```

## Phase 2: Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Or in Google Colab:

```python
!pip install -U ultralytics pandas scikit-learn pyyaml matplotlib
```

## Phase 3: Training

If your dataset is already split into YOLO format, train directly:

```bash
python scripts/train_yolov8.py \
  --data /content/vehicle_dataset/data.yaml \
  --project runs/vehicle_detection \
  --epochs 100 \
  --imgsz 640
```

This trains:

- `runs/vehicle_detection/vehicle_yolov8s/weights/best.pt`
- `runs/vehicle_detection/vehicle_yolov8m/weights/best.pt`
- `runs/vehicle_detection/vehicle_yolov8l/weights/best.pt`

If GPU memory is limited, reduce batch sizes inside `scripts/train_yolov8.py`.

## Optional: Split a Raw Dataset

If you have unsplit image and label folders:

```bash
python scripts/prepare_dataset.py \
  --images /content/raw_vehicle_images \
  --labels /content/raw_vehicle_labels \
  --output /content/vehicle_dataset
```

This creates the required `images/train`, `images/val`, `images/test`, `labels/train`, `labels/val`, `labels/test`, and `data.yaml`.

## Phase 4: Evaluation and Comparison

Evaluate the three trained models on the test set:

```bash
python scripts/evaluate_models.py \
  --data /content/vehicle_dataset/data.yaml \
  --project runs/vehicle_detection \
  --output reports/yolov8_vehicle_comparison.csv
```

The CSV report includes:

- `mAP@0.5`
- `mAP@0.5:0.95`
- precision
- recall
- inference time
- model size

## Suggested Academic Discussion

Use `mAP@0.5:0.95` as the main accuracy metric because it averages performance across multiple IoU thresholds. Compare it with inference time and model size:

- If YOLOv8s is much faster but only slightly less accurate, it is better for real-time deployment.
- If YOLOv8l gives the best mAP but is slow, it is better for offline or high-accuracy applications.
- YOLOv8m often provides the best practical compromise.

## Notebook

The complete notebook is available at:

```text
notebooks/vehicle_yolov8_project.ipynb
```

Use it directly in Jupyter Notebook or upload it to Google Colab.
