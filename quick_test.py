"""Quick sanity test - verify pipeline works."""

import cv2
import time
import numpy as np
from config import load_config
from main import create_pipeline_from_config

# Load config and initialize pipeline
print("Loading config...")
config = load_config('configs/config.yaml')
print("Initializing pipeline...")
pipeline = create_pipeline_from_config(config)

# Load test image
print("Loading test image...")
test_image = cv2.imread('test.jpg')  # Or use a default test image
if test_image is None:
    # Create dummy test image if file doesn't exist
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print("Using dummy test image (test.jpg not found)")

# Run pipeline
print("Running pipeline...")
start_time = time.time()
result = pipeline.process_frame(test_image, frame_id=0)
processing_time = time.time() - start_time

# Print results
print("\n" + "="*50)
print("QUICK TEST RESULTS")
print("="*50)
print(f"Processing time: {processing_time*1000:.2f}ms")
print(f"Person ID: {result.get('person_id', 'None')}")
print(f"Confidence: {result.get('confidence', 0.0):.4f}")
print(f"Frame ID: {result.get('frame_id', 0)}")
print("="*50)

# Display image with results
if result.get('person_id'):
    # Draw text if person detected
    person_id = result['person_id']
    confidence = result.get('confidence', 0.0)
    text = f"{person_id} ({confidence:.2f})"
    cv2.putText(test_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.putText(test_image, f"Time: {processing_time*1000:.1f}ms", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
cv2.imshow('Quick Test', test_image)
print("\nPress any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n✓ Quick test completed successfully!")

