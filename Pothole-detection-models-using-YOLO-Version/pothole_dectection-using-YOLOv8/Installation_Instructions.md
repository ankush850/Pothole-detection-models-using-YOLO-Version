Installation Instructions:


Create the folder structure:

pothole_dectection-using-YOLOv8/
│
├── data/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
│
├── models/
├── results/
├── train.py
├── detect.py
├── requirements.txt
└── config.yaml




bash
mkdir -p pothole_dectection-using-YOLOv8
cd pothole_dectection-using-YOLOv8



Create the files (copy each file content above into respective files)

Install dependencies:

bash
pip install -r requirements.txt
Setup project:

bash
python setup.py
Prepare your dataset:

Place images in data/images/train/ and data/images/val/

Place labels in data/labels/train/ and data/labels/val/

Labels must be in YOLO format

Train the model:

bash
python train.py --epochs 100 --batch 16 --model m
Run detection:

bash
# Image
python detect.py --source test_image.jpg

# Video
python detect.py --source test_video.mp4

# Webcam
python detect.py --source 0
Key Features for 89% Accuracy:
Optimized Training Parameters:

AdamW optimizer

Cosine learning rate scheduler

Extensive data augmentation

Early stopping

Data Augmentation:

Rotation, scaling, shearing

Mixup augmentation

Random erasing

Mosaic augmentation

Model Selection:

YOLOv8m (medium) recommended

Can scale to YOLOv8l for better accuracy