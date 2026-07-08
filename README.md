# ML-Inference-Bench

I've been interested in NVIDIA and roles working on deep neural networks for
a while, and most of my other projects are about building and training
models, not the part that actually matters once a model ships: making it
fast and cheap enough to run in production. TensorRT is the tool
NVIDIA-adjacent teams actually use for that, so I built ML-Inference-Bench
to get real, hands-on experience with it instead of just reading docs:
exporting models to ONNX, building engines at different precisions, and
measuring what quantizing down to INT8 actually costs you in accuracy
versus what it saves you in latency.

---

## What it does

A CLI benchmarking framework that takes a pretrained PyTorch model, exports
it to ONNX, builds a TensorRT engine at FP32, FP16, or INT8 precision, and
measures latency (p50/p95/p99 via CUDA events), throughput, GPU memory, and
accuracy drift versus the original model. Supports ResNet50, EfficientNet,
YOLOv8, BERT-base, and Whisper-tiny out of the box.

---


## Pipeline

```
PyTorch model
      |
      v
export to ONNX
      |
      v
optimize with TensorRT (FP32 / FP16 / INT8)
      |
      v
benchmark: latency, throughput, memory, accuracy
      |
      v
export report (JSON / CSV / HTML)
```

---

## Key Features

- **Multi-precision optimization**: FP32, FP16, INT8 TensorRT engine
  building, including a custom INT8 entropy calibrator with calibration
  caching
- **GPU-accurate latency measurement**: CUDA event-based timing with a
  100-run warmup before the 1000-run measurement window, reported as
  p50/p95/p99/mean
- **Model zoo**: ResNet50, EfficientNet, YOLOv8, BERT-base, Whisper-tiny
- **Exportable reports**: JSON, CSV, and a styled static HTML report per run

## Usage

```bash
python main.py --model resnet50 --precision fp16 --report html
python main.py --all-models --precision int8 --report json
```

Requires an NVIDIA GPU with CUDA and TensorRT installed. This isn't
runnable on a laptop without a GPU. That's a real constraint of the
problem, not a gap in the code.

## Stack

| What | How |
|------|-----|
| Export | PyTorch to ONNX |
| Optimization | NVIDIA TensorRT (FP32/FP16/INT8) |
| Latency measurement | CUDA events, PyCUDA |
| Reports | pandas to JSON/CSV/HTML |

## What I'd build next

- Run this against a real GPU box and publish actual numbers here instead
  of leaving the results section for whoever runs it locally
- Extend accuracy measurement beyond top-1 to something like mAP for
  YOLOv8 specifically, since classification accuracy isn't the right
  metric for a detection model
