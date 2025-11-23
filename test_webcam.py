"""Simple webcam testing script for face recognition pipeline."""

import cv2
import time
from dual_stream_face_recognition.pipeline import DualStreamPipeline
from dual_stream_face_recognition.detection.yolo_detector import YOLODetector
from dual_stream_face_recognition.buffer.adaptive_buffer import AdaptiveBuffer
from dual_stream_face_recognition.voting.temporal_voting import TemporalVotingAlgorithm

# Initialize pipeline
yolo = YOLODetector(device='cpu')
buffer = AdaptiveBuffer(buffer_size=30)
voting = TemporalVotingAlgorithm(window_size=8)
pipeline = DualStreamPipeline(yolo_detector=yolo, buffer=buffer, voting=voting)

# Open webcam
cap = cv2.VideoCapture(0)

frame_id = 0
fps_start = time.time()
fps_count = 0
current_fps = 0.0

print("Controls: 'q'=quit, 's'=save")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    result = pipeline.process_frame(frame, frame_id=frame_id)
    
    # Draw person_id and confidence
    if result.get('person_id'):
        person_id = result['person_id']
        confidence = result.get('confidence', 0.0)
        text = f"{person_id} ({confidence:.2f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw FPS counter on screen
    fps_text = f"FPS: {current_fps:.1f}"
    cv2.putText(frame, fps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Draw buffer status
    buffer_count = len(buffer) if buffer else 0
    buffer_text = f"Buffer: {buffer_count}/{buffer.buffer_size if buffer else 0}"
    cv2.putText(frame, buffer_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Draw confidence score
    confidence = result.get('confidence', 0.0)
    conf_text = f"Confidence: {confidence:.2f}"
    cv2.putText(frame, conf_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow('Face Recognition - Webcam', frame)
    frame_id += 1
    fps_count += 1
    
    # Update FPS every 30 frames
    if fps_count % 30 == 0:
        elapsed = time.time() - fps_start
        current_fps = fps_count / elapsed if elapsed > 0 else 0
        fps_start = time.time()
        fps_count = 0
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(f"webcam_frame_{frame_id}.jpg", frame)
        print(f"Saved: webcam_frame_{frame_id}.jpg")

cap.release()
cv2.destroyAllWindows()
print("Done!")

