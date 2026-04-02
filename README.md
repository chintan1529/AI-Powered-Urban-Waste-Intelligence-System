# Urban Waste Intelligence System: SOTA Object Detection Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8x-Computer_Vision-orange.svg)](https://github.com/ultralytics/ultralytics)
[![ONNX](https://img.shields.io/badge/ONNX-Edge_Optimized-lightgrey.svg)](https://onnx.ai/)
[![Gradio](https://img.shields.io/badge/Gradio-Web_UI-ff69b4.svg)](https://gradio.app/)

## Abstract
The Urban Waste Intelligence System is an end-to-end machine learning pipeline engineered to detect, classify, and isolate municipal street waste in real-time. Built to bridge the gap between raw computer vision inference and actionable municipal telemetry, this system utilizes a heavily augmented YOLOv8x (Extra-Large) architecture optimized for extreme environmental variables and edge-device deployment.

## Key Engineering Features

* **State-of-the-Art (SOTA) Architecture:** Utilizes the YOLOv8x model (68M parameters) to maximize recall and precision on small, occluded, or deformed objects typical of urban waste.
* **Hyper-Augmented Training Protocol:** Implements advanced dataset augmentation to prevent overfitting, including Mosaic stitching (1.0), MixUp blending (0.2), and dynamic HSV exposure/saturation shifting to simulate varying lighting and weather conditions.
* **Optimized Convergence Strategy:** Replaces static learning rates with a Cosine Annealing Learning Rate Scheduler (`cos_lr=True`) coupled with the AdamW optimizer and a 50-epoch patience early-stopping trigger.
* **Edge-Deployment Optimization:** Compiles the final PyTorch weights (`.pt`) into an FP16 Half-Precision ONNX format (`.onnx`), stripping training metadata to accelerate inference speeds by up to 3x on edge hardware (e.g., NVIDIA Jetson Nano).
* **Interactive Inference Dashboard:** Features a decoupled Gradio web interface with an integrated taxonomy engine, automatically sanitizing transparent alpha channels (RGBA to RGB) and mapping raw detection IDs to a professional waste classification schema.

## Repository Structure

```text
urban-waste-intelligence/
│
├── models/
│   ├── best.pt                 # Native PyTorch weights
│   └── best.onnx               # FP16 Edge-optimized weights
│
├── src/
│   ├── app.py                  # Web application and inference engine
│   └── train.py                # SOTA hyper-training script
│
├── requirements.txt            # System dependencies
├── .gitignore                  # Git exclusion parameters
└── README.md                   # Project documentation
