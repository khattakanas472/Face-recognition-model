"""Simple video testing script for face recognition pipeline."""

import cv2
import time
import sys
from dual_stream_face_recognition.pipeline import DualStreamPipeline
from dual_stream_face_recognition.detection.yolo_detector import YOLODetector
from dual_stream_face_recognition.buffer.adaptive_buffer import AdaptiveBuffer
from dual_stream_face_recognition.voting.temporal_voting import TemporalVotingAlgorithm

# Initialize pipeline
yolo = YOLODetector(device='cpu')
buffer = AdaptiveBuffer(buffer_size=30)
voting = TemporalVotingAlgorithm(window_size=8)
pipeline = DualStreamPipeline(yolo_detector=yolo, buffer=buffer, voting=voting)

# Open video
video_path = sys.argv[1] if len(sys.argv) > 1 else 'test.mp4'
cap = cv2.VideoCapture(video_path)

frame_id = 0
paused = False
fps_start = time.time()
fps_count = 0

print("Controls: 'q'=quit, 's'=save, SPACE=pause/resume")

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = pipeline.process_frame(frame, frame_id=frame_id)
        
        if result.get('person_id'):
            text = f"{result['person_id']} ({result.get('confidence', 0.0):.2f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Face Recognition', frame)
        frame_id += 1
        fps_count += 1
        
        if fps_count % 30 == 0:
            fps = fps_count / (time.time() - fps_start)
            print(f"FPS: {fps:.1f}")
            fps_start = time.time()
            fps_count = 0
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(f"frame_{frame_id}.jpg", frame)
        print(f"Saved: frame_{frame_id}.jpg")
    elif key == ord(' '):
        paused = not paused

cap.release()
cv2.destroyAllWindows()
print("Done!")

