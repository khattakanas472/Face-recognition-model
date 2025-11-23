"""Simple evaluation script for face recognition accuracy."""

import numpy as np
from pathlib import Path
from dataset import SimpleFaceDataset
from dual_stream_face_recognition.pipeline import DualStreamPipeline
from dual_stream_face_recognition.detection.yolo_detector import YOLODetector
from dual_stream_face_recognition.buffer.adaptive_buffer import AdaptiveBuffer
from dual_stream_face_recognition.voting.temporal_voting import TemporalVotingAlgorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate(test_dir, gallery_path='gallery.npy', save_path='results.txt'):
    """Evaluate pipeline accuracy on test dataset."""
    # Load test dataset
    test_dataset = SimpleFaceDataset(test_dir)
    
    # Initialize pipeline
    yolo = YOLODetector(device='cpu')
    buffer = AdaptiveBuffer(buffer_size=30)
    voting = TemporalVotingAlgorithm(window_size=8)
    
    # Load gallery if available
    gallery_embeddings = None
    gallery_ids = []
    if Path(gallery_path).exists():
        from build_gallery import load_gallery
        import torch
        gallery = load_gallery(gallery_path)
        if gallery:
            gallery_embeddings = torch.stack([torch.from_numpy(emb) for emb in gallery.values()])
            gallery_ids = list(gallery.keys())
    
    pipeline = DualStreamPipeline(
        yolo_detector=yolo,
        buffer=buffer,
        voting=voting,
        gallery_embeddings=gallery_embeddings,
        gallery_ids=gallery_ids
    )
    
    # Run evaluation
    true_labels = []
    pred_labels = []
    
    print(f"Evaluating {len(test_dataset)} images...")
    
    for idx, (img_tensor, true_id) in enumerate(test_dataset):
        # Convert tensor to numpy for pipeline
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_bgr = np.flip(img_np, axis=2)  # RGB to BGR
        
        # Process through pipeline
        result = pipeline.process_frame(img_bgr, frame_id=idx)
        pred_id = result.get('person_id')
        
        true_labels.append(true_id)
        pred_labels.append(pred_id if pred_id else 'unknown')
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(test_dataset)} images...")
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    
    # Classification report
    report = classification_report(true_labels, pred_labels, zero_division=0)
    
    # Confusion matrix
    all_labels = sorted(set(true_labels + pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nTotal images: {len(test_dataset)}")
    print(f"Correct predictions: {sum(1 for t, p in zip(true_labels, pred_labels) if t == p)}")
    
    print("\n" + "-"*60)
    print("CLASSIFICATION REPORT")
    print("-"*60)
    print(report)
    
    print("\n" + "-"*60)
    print("CONFUSION MATRIX")
    print("-"*60)
    print(f"{'True\\Pred':<15}", end="")
    for label in all_labels:
        print(f"{label[:10]:<12}", end="")
    print()
    
    for i, true_label in enumerate(all_labels):
        print(f"{true_label[:14]:<15}", end="")
        for j, pred_label in enumerate(all_labels):
            print(f"{cm[i][j]:<12}", end="")
        print()
    
    # Save to file
    with open(save_path, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"\nTotal images: {len(test_dataset)}\n")
        f.write(f"Correct predictions: {sum(1 for t, p in zip(true_labels, pred_labels) if t == p)}\n")
        f.write("\n" + "-"*60 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("-"*60 + "\n")
        f.write(report)
        f.write("\n" + "-"*60 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("-"*60 + "\n")
        f.write(f"{'True\\Pred':<15}")
        for label in all_labels:
            f.write(f"{label[:10]:<12}")
        f.write("\n")
        for i, true_label in enumerate(all_labels):
            f.write(f"{true_label[:14]:<15}")
            for j, pred_label in enumerate(all_labels):
                f.write(f"{cm[i][j]:<12}")
            f.write("\n")
    
    print(f"\nResults saved to: {save_path}")
    
    return accuracy, report, cm


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <test_dataset_dir> [gallery_path] [results_path]")
        sys.exit(1)
    
    test_dir = sys.argv[1]
    gallery_path = sys.argv[2] if len(sys.argv) > 2 else 'gallery.npy'
    results_path = sys.argv[3] if len(sys.argv) > 3 else 'results.txt'
    
    evaluate(test_dir, gallery_path, results_path)

