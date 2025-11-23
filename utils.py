"""Basic utility functions for face recognition."""

import cv2
import numpy as np
import psutil


def draw_bbox(frame, bbox, person_id, confidence):
    """Draw bounding box with color based on confidence."""
    x1, y1, x2, y2 = map(int, bbox)
    color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.6 else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{person_id} ({confidence:.2f})"
    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def calculate_fps(frame_times):
    """Calculate rolling average FPS from timestamps."""
    if len(frame_times) < 2:
        return 0.0
    intervals = np.diff(frame_times[-30:])  # Last 30 frames
    return 1.0 / np.mean(intervals) if len(intervals) > 0 and np.mean(intervals) > 0 else 0.0


def resize_with_aspect(image, target_size):
    """Resize keeping aspect ratio, pad with black if needed."""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    padded = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return padded


def get_memory_usage():
    """Return current RAM usage in MB."""
    return psutil.virtual_memory().used / (1024 * 1024)

