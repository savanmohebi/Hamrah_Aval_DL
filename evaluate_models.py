{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Vehicle Detection Using YOLOv8\n",
    "\n",
    "This notebook fine-tunes YOLOv8 Small, Medium, and Large models for 5 vehicle classes: `Car`, `Bicycle`, `Bus`, `Truck`, and `Motorcycle`."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Phase 1: Dataset Preparation\n",
    "\n",
    "Use Roboflow, CVAT, or LabelImg to draw bounding boxes. Export labels in YOLO format. Each label file must have rows formatted as:\n",
    "\n",
    "```txt\n",
    "class_id x_center y_center width height\n",
    "```\n",
    "\n",
    "Class mapping:\n",
    "\n",
    "```txt\n",
    "0 Car\n",
    "1 Bicycle\n",
    "2 Bus\n",
    "3 Truck\n",
    "4 Motorcycle\n",
    "```"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pathlib import Path\n",
    "import yaml\n",
    "\n",
    "DATASET_DIR = Path('/content/vehicle_dataset')\n",
    "DATASET_DIR.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "data_yaml = {\n",
    "    'path': str(DATASET_DIR),\n",
    "    'train': 'images/train',\n",
    "    'val': 'images/val',\n",
    "    'test': 'images/test',\n",
    "    'names': {\n",
    "        0: 'Car',\n",
    "        1: 'Bicycle',\n",
    "        2: 'Bus',\n",
    "        3: 'Truck',\n",
    "        4: 'Motorcycle',\n",
    "    }\n",
    "}\n",
    "\n",
    "with open(DATASET_DIR / 'data.yaml', 'w', encoding='utf-8') as f:\n",
    "    yaml.safe_dump(data_yaml, f, sort_keys=False)\n",
    "\n",
    "print((DATASET_DIR / 'data.yaml').read_text())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Expected folder structure:\n",
    "\n",
    "```txt\n",
    "vehicle_dataset/\n",
    "├── data.yaml\n",
    "├── images/train, images/val, images/test\n",
    "└── labels/train, labels/val, labels/test\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Phase 2: Environment Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install -U ultralytics pandas pyyaml matplotlib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import pandas as pd\n",
    "from ultralytics import YOLO\n",
    "import ultralytics\n",
    "\n",
    "ultralytics.checks()\n",
    "device = 0 if torch.cuda.is_available() else 'cpu'\n",
    "print('Device:', device)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Phase 3: Training Phase"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pathlib import Path\n",
    "\n",
    "DATA_YAML = str(DATASET_DIR / 'data.yaml')\n",
    "PROJECT_DIR = '/content/yolo_vehicle_runs'\n",
    "\n",
    "EPOCHS = 100\n",
    "IMG_SIZE = 640\n",
    "PATIENCE = 20\n",
    "SEED = 42\n",
    "\n",
    "model_configs = {\n",
    "    'small': {'weights': 'yolov8s.pt', 'batch': 16, 'run_name': 'vehicle_yolov8s'},\n",
    "    'medium': {'weights': 'yolov8m.pt', 'batch': 8, 'run_name': 'vehicle_yolov8m'},\n",
    "    'large': {'weights': 'yolov8l.pt', 'batch': 4, 'run_name': 'vehicle_yolov8l'},\n",
    "}\n",
    "\n",
    "trained_models = {}\n",
    "\n",
    "for scale_name, config in model_configs.items():\n",
    "    print(f'\\n========== Training YOLOv8-{scale_name} ==========')\n",
    "    model = YOLO(config['weights'])\n",
    "    model.train(\n",
    "        data=DATA_YAML,\n",
    "        epochs=EPOCHS,\n",
    "        imgsz=IMG_SIZE,\n",
    "        batch=config['batch'],\n",
    "        patience=PATIENCE,\n",
    "        project=PROJECT_DIR,\n",
    "        name=config['run_name'],\n",
    "        seed=SEED,\n",
    "        pretrained=True,\n",
    "        optimizer='auto',\n",
    "        amp=True,\n",
    "        val=True,\n",
    "        plots=True,\n",
    "        device=device,\n",
    "    )\n",
    "    trained_models[scale_name] = Path(PROJECT_DIR) / config['run_name'] / 'weights' / 'best.pt'\n",
    "    print('Best weights:', trained_models[scale_name])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Phase 4: Evaluation and Comparison"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "\n",
    "comparison_rows = []\n",
    "\n",
    "for scale_name, weights_path in trained_models.items():\n",
    "    print(f'\\n========== Evaluating YOLOv8-{scale_name} ==========')\n",
    "    model = YOLO(str(weights_path))\n",
    "    metrics = model.val(\n",
    "        data=DATA_YAML,\n",
    "        split='test',\n",
    "        imgsz=IMG_SIZE,\n",
    "        batch=1,\n",
    "        conf=0.001,\n",
    "        iou=0.6,\n",
    "        plots=True,\n",
    "        verbose=False,\n",
    "    )\n",
    "\n",
    "    comparison_rows.append({\n",
    "        'model': f'YOLOv8-{scale_name}',\n",
    "        'weights': str(weights_path),\n",
    "        'mAP@0.5': float(metrics.box.map50),\n",
    "        'mAP@0.5:0.95': float(metrics.box.map),\n",
    "        'precision': float(metrics.box.mp),\n",
    "        'recall': float(metrics.box.mr),\n",
    "        'model_size_MB': os.path.getsize(weights_path) / (1024 * 1024),\n",
    "        'preprocess_ms': metrics.speed.get('preprocess'),\n",
    "        'inference_ms': metrics.speed.get('inference'),\n",
    "        'postprocess_ms': metrics.speed.get('postprocess'),\n",
    "    })\n",
    "\n",
    "comparison_df = pd.DataFrame(comparison_rows)\n",
    "comparison_df.to_csv('/content/yolov8_vehicle_model_comparison.csv', index=False)\n",
    "comparison_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "comparison_df.sort_values('mAP@0.5:0.95', ascending=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Report Interpretation\n",
    "\n",
    "- YOLOv8s is usually fastest and smallest.\n",
    "- YOLOv8m is usually the best balance.\n",
    "- YOLOv8l is usually most accurate but slower and heavier.\n",
    "- Use `mAP@0.5:0.95` as the main final detection metric.\n",
    "- Use inference time and model size to discuss deployment trade-offs."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.x"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
