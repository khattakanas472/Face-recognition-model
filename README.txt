Dual-Stream Face Recognition - Research Code

Setup:

1. pip install -r requirements.txt

2. python scripts/setup_models.py

3. python scripts/build_gallery.py --dataset /path/to/faces

Usage:

- Test video: python main.py --mode video --input test.mp4

- Test webcam: python main.py --mode webcam

- Evaluate: python main.py --mode eval

Experiments:

- Ablation: python scripts/ablation.py

- Comparison: python scripts/compare_methods.py

