# ML-Inference-Bench
ML-Inference-Bench is a benchmarking framework for evaluating neural network inference performance using NVIDIA TensorRT. It systematically compares FP32, FP16, and INT8 precision modes across latency, throughput, memory usage, and predictive accuracy.
The goal of this project is to provide a reproducible environment for studying the tradeoffs between model precision and inference efficiency, enabling informed decisions when deploying machine learning models in production environments.

## Overview

In production, model efficiency matters as much as accuracy. ML-Inference-Bench provides:

- Automated TensorRT optimization for FP32, FP16, and INT8  
- Reproducible benchmarking of latency, throughput, memory, and accuracy  
- Visualization dashboard for easy performance analysis  
- Preconfigured model zoo for quick experimentation  

---

## Key Features

- **Multi-Precision Optimization**: FP32, FP16, INT8 TensorRT engine building  
- **Reproducible Benchmarking**: Latency, throughput, and GPU memory metrics using CUDA Events  
- **Visualization Dashboard**: Interactive Plotly Dash dashboard for performance analysis  
- **Model Zoo**: Supports ResNet50, EfficientNet, YOLOv8, BERT-base, Whisper-tiny  
- **Exportable Reports**: JSON, CSV, HTML formats for sharing and analysis  

## Pipeline
PyTorch model
      ↓
export to ONNX
      ↓
optimize with TensorRT
      ↓
benchmark inference
      ↓
visualize performance tradeoffs