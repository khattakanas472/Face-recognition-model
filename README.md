# Dual-Stream Face Recognition Research Framework

A high-performance machine learning library for face recognition that combines lightweight detection with high-precision verification through an adaptive buffering and temporal voting system.

## Overview

This framework implements a dual-stream architecture that optimizes both speed and accuracy:

1. **Lightweight Detection Stream**: Uses YOLOv8-Nano for fast face detection
2. **High-Precision Verification Stream**: Uses RetinaFace for accurate face verification
3. **Adaptive Buffer**: Intelligent frame management for optimal performance
4. **Temporal Voting**: Consensus mechanism for robust recognition

## Features

- 🚀 **Fast Detection**: YOLOv8-Nano for real-time face detection
- 🎯 **High Accuracy**: RetinaFace for precise face verification
- 📊 **Adaptive Buffering**: Smart frame management
- 🗳️ **Temporal Voting**: Consensus-based recognition
- 🔧 **Research-Friendly**: Modular design for experimentation

## Project Structure

```
dual_stream_face_recognition/
├── dual_stream_face_recognition/
│   ├── __init__.py
│   ├── detection/          # YOLOv8-Nano detection module
│   │   ├── __init__.py
│   │   └── yolov8_detector.py
│   ├── verification/       # RetinaFace verification module
│   │   ├── __init__.py
│   │   └── retinaface_verifier.py
│   ├── buffer/             # Adaptive buffer module
│   │   ├── __init__.py
│   │   └── adaptive_buffer.py
│   ├── voting/             # Temporal voting module
│   │   ├── __init__.py
│   │   └── temporal_voting.py
│   ├── utils/              # Utility functions
│   │   ├── __init__.py
│   │   └── image_utils.py
│   └── config/             # Configuration settings
│       ├── __init__.py
│       └── settings.py
├── setup.py
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for optimal performance)
- pip or conda

### Install from Source

1. Clone the repository:
```bash
git clone <repository-url>
cd dual-stream-face-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

## Usage

*(Usage examples will be added once the models are implemented)*

## Development

### Running Tests

*(Test instructions will be added once tests are implemented)*

### Code Style

This project follows PEP 8 style guidelines. Format code using:

```bash
black dual_stream_face_recognition/
```

## Dependencies

### Core
- **PyTorch** (>=2.0.0): Deep learning framework
- **OpenCV** (>=4.8.0): Computer vision operations
- **NumPy** (>=1.24.0): Numerical computations

### Models
- **Ultralytics** (>=8.0.0): YOLOv8 implementation
- **InsightFace** (>=0.7.3): RetinaFace implementation

### Utilities
- **SciPy**, **scikit-learn**: Scientific computing
- **Matplotlib**: Visualization
- **tqdm**: Progress bars
- **PyYAML**: Configuration management
- **Loguru**: Logging

## License

*(License information to be added)*

## Contributing

*(Contributing guidelines to be added)*

## Citation

*(Citation information to be added once research is published)*

## Authors

Research Team

## Acknowledgments

- YOLOv8 by Ultralytics
- RetinaFace by InsightFace

