"""Main entry point for face recognition pipeline."""

import argparse
from config import load_config
from dual_stream_face_recognition.pipeline import DualStreamPipeline
from dual_stream_face_recognition.detection.yolo_detector import YOLODetector
from dual_stream_face_recognition.detection.retinaface_detector import RetinaFaceDetector
from dual_stream_face_recognition.buffer.adaptive_buffer import AdaptiveBuffer
from dual_stream_face_recognition.voting.temporal_voting import TemporalVotingAlgorithm
import torch


def run_video(pipeline, video_path):
    """Run pipeline on video file."""
    import cv2
    import time
    from utils import calculate_fps
    
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    frame_times = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = pipeline.process_frame(frame, frame_id=frame_id)
        frame_times.append(time.time())
        
        if result.get('person_id'):
            person_id = result['person_id']
            confidence = result.get('confidence', 0.0)
            text = f"{person_id} ({confidence:.2f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        fps = calculate_fps(frame_times)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow('Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_id += 1
    
    cap.release()
    cv2.destroyAllWindows()


def run_webcam(pipeline):
    """Run pipeline on webcam."""
    import cv2
    import time
    from utils import calculate_fps
    
    cap = cv2.VideoCapture(0)
    frame_id = 0
    frame_times = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = pipeline.process_frame(frame, frame_id=frame_id)
        frame_times.append(time.time())
        
        if result.get('person_id'):
            person_id = result['person_id']
            confidence = result.get('confidence', 0.0)
            text = f"{person_id} ({confidence:.2f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        fps = calculate_fps(frame_times)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow('Face Recognition - Webcam', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_id += 1
    
    cap.release()
    cv2.destroyAllWindows()


def run_evaluation(pipeline, test_dir, gallery_path='gallery.npy'):
    """Run evaluation on test dataset."""
    from evaluate import evaluate
    evaluate(test_dir, gallery_path)


def create_pipeline_from_config(config):
    """Initialize pipeline from config."""
    # Set CPU threads
    torch.set_num_threads(config['inference'].get('num_threads', 2))
    
    # Initialize detectors
    yolo = YOLODetector(
        device=config['inference'].get('device', 'cpu'),
        confidence_threshold=config['inference'].get('confidence_threshold', 0.5)
    ) if config['models'].get('yolo_path') else None
    
    retina = RetinaFaceDetector(
        device=config['inference'].get('device', 'cpu'),
        confidence_threshold=config['inference'].get('confidence_threshold', 0.5)
    ) if config['models'].get('retina_path') else None
    
    # Initialize buffer
    buffer = AdaptiveBuffer(
        max_memory_mb=config['buffer'].get('max_memory_mb', 512),
        max_age_seconds=config['buffer'].get('max_age_sec', 60),
        buffer_size=30
    )
    
    # Initialize voting
    voting = TemporalVotingAlgorithm(
        window_size=config['fusion'].get('window_size', 8),
        yolo_weight=config['fusion'].get('yolo_weight', 1.0),
        retina_weight=config['fusion'].get('retina_weight', 1.5)
    )
    
    # Create pipeline
    pipeline = DualStreamPipeline(
        yolo_detector=yolo,
        retina_detector=retina,
        buffer=buffer,
        voting=voting,
        batch_size=config['inference'].get('batch_size', 8)
    )
    
    return pipeline


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['video', 'webcam', 'eval'], required=True)
    parser.add_argument('--input', help='video path for video mode or test dataset dir for eval mode')
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--gallery', default='gallery.npy', help='gallery path for eval mode')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Initialize pipeline
    pipeline = create_pipeline_from_config(config)
    
    # Run mode
    if args.mode == 'video':
        if not args.input:
            print("Error: --input required for video mode")
            return
        run_video(pipeline, args.input)
    elif args.mode == 'webcam':
        run_webcam(pipeline)
    elif args.mode == 'eval':
        if not args.input:
            print("Error: --input required for eval mode (test dataset directory)")
            return
        run_evaluation(pipeline, args.input, args.gallery)


if __name__ == '__main__':
    main()

