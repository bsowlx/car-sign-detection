# Traffic Sign Detection using YOLOv11n

## Overview
This project implements a traffic sign detection system using the YOLOv11n architecture. The model has been trained on a custom dataset of traffic signs and can detect various classes of signs in images and videos.

## Features
- **Real-time Detection**: Detects traffic signs in uploaded images and videos.
- **Interactive UI**: Built with Streamlit for an easy-to-use web interface.
- **Custom Model**: Trained specifically on the provided car/traffic sign dataset.

## Installation

1. Ensure you have Python installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Web Interface
To start the Streamlit application, run:
```bash
streamlit run app.py
```
This will open a web browser where you can upload images or videos for detection.

### Generating Training Reports
To generate visualization plots of the training process (Loss and mAP curves):
```bash
python generate_report.py
```
The images will be saved in the `report_images/` directory.

## Model Performance

### Training Configuration
- **Model**: YOLOv11n
- **Architecture**: 181 layers, 2,592,765 parameters
- **Epochs**: 30
- **Batch Size**: 57 (auto-computed)
- **Image Size**: 640x640
- **Optimizer**: AdamW (lr=0.000526, momentum=0.9)
- **Device**: CUDA (Tesla T4, 15095MiB)
- **Training Time**: 0.536 hours (~32 minutes)
- **Dataset Size**: 3,530 training images, 801 validation images

### Final Metrics (Best Model - Epoch 26)
| Metric | Value |
|--------|-------|
| Precision (P) | 0.917 |
| Recall (R) | 0.904 |
| mAP@50 | 0.950 |
| mAP@50-95 | 0.825 |

### Per-Class Performance
| Class | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|--------|-----------|-----------|--------|--------|-----------|
| Green Light | 87 | 122 | 0.859 | 0.750 | 0.832 | 0.500 |
| Red Light | 74 | 108 | 0.861 | 0.778 | 0.833 | 0.535 |
| Speed Limit 100 | 52 | 52 | 0.975 | 0.942 | 0.982 | 0.896 |
| Speed Limit 110 | 17 | 17 | 0.676 | 0.859 | 0.910 | 0.827 |
| Speed Limit 120 | 60 | 60 | 0.987 | 0.967 | 0.993 | 0.900 |
| Speed Limit 20 | 56 | 56 | 1.000 | 0.964 | 0.986 | 0.877 |
| Speed Limit 30 | 71 | 74 | 0.934 | 0.946 | 0.980 | 0.927 |
| Speed Limit 40 | 53 | 55 | 0.930 | 0.960 | 0.990 | 0.878 |
| Speed Limit 50 | 68 | 71 | 0.965 | 0.915 | 0.964 | 0.867 |
| Speed Limit 60 | 76 | 76 | 0.884 | 0.908 | 0.964 | 0.880 |
| Speed Limit 70 | 78 | 78 | 0.954 | 0.962 | 0.981 | 0.907 |
| Speed Limit 80 | 56 | 56 | 0.947 | 0.929 | 0.980 | 0.866 |
| Speed Limit 90 | 38 | 38 | 0.888 | 0.789 | 0.906 | 0.767 |
| Stop | 81 | 81 | 0.974 | 0.988 | 0.994 | 0.927 |

### Training Progress
| Epoch | Box Loss | Cls Loss | DFL Loss | mAP@50 | mAP@50-95 |
|-------|----------|----------|----------|--------|-----------|
| 1 | 0.9053 | 3.857 | 1.225 | 0.152 | 0.123 |
| 5 | 0.6989 | 1.586 | 1.050 | 0.686 | 0.571 |
| 10 | 0.6278 | 0.994 | 1.014 | 0.865 | 0.727 |
| 15 | 0.5782 | 0.7902 | 0.988 | 0.913 | 0.771 |
| 20 | 0.5494 | 0.669 | 0.974 | 0.937 | 0.800 |
| 25 | 0.5315 | 0.4338 | 0.931 | 0.941 | 0.812 |
| 30 | 0.4896 | 0.374 | 0.9097 | 0.950 | 0.826 |

### Inference Speed
- **Preprocess**: 0.2ms per image
- **Inference**: 2.5ms per image
- **Postprocess**: 4.7ms per image
- **Total**: ~7.4ms per image (~135 FPS)

## Dataset
The dataset is organized in the `car/` directory, following the YOLO format with `train`, `valid`, and `test` splits.
