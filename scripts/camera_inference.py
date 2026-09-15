import os
import sys
import time
import uuid
import argparse
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fedtrap.camera import create_camera, MotionDetector
from fedtrap.inference import InsectClassifier
from fedtrap.temporal_filter import TemporalFilter
from fedtrap.decision_policy import get_decision
from fedtrap.servo_controller import create_servo
from fedtrap.logging_utils import EventLogger
from fedtrap.config import get_config

def main():
    parser = argparse.ArgumentParser(description="Camera Inference Pipeline")
    parser.add_argument("--source", type=str, required=True, help="image_path / video_path / webcam / picamera")
    parser.add_argument("--model", type=str, default="models/classifier_best.pt", help="Path to model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config")
    parser.add_argument("--threshold", type=float, default=0.85, help="Confidence threshold")
    parser.add_argument("--display", action="store_true", help="Display video stream with annotations")
    parser.add_argument("--save-log", action="store_true", help="Log events to CSV")
    args = parser.parse_args()

    cfg = get_config()
    
    # Init components
    print(f"Initializing camera source: {args.source}")
    camera = create_camera(args.source)
    motion_det = MotionDetector(threshold=25, min_area=500)
    
    print("Loading classifier...")
    classifier = InsectClassifier(model_path=args.model, config_path=args.config)
    
    temporal_filter = TemporalFilter(confirmation_frames=cfg.get('decision', {}).get('confirmation_frames', 3),
                                     hold_duration=cfg.get('decision', {}).get('hold_duration_sec', 2.0))
    
    servo = create_servo(cfg.get('servo', {}))
    
    if args.save_log:
        logger = EventLogger(csv_path="results/inference_log.csv")
    else:
        logger = None

    is_pi = (args.source == 'picamera' or str(servo.__class__.__name__) == 'PiServoController')
    print(f"Mode: {'PI' if is_pi else 'DESKTOP'}")

    print("Starting pipeline. Press Ctrl+C to stop.")
    
    try:
        frames_count = 0
        start_time = time.time()
        
        while camera.is_opened():
            ret, frame = camera.read()
            if not ret or frame is None:
                if args.source.endswith(('.mp4', '.avi')):
                    break
                time.sleep(0.01)
                continue
                
            frames_count += 1
            motion, thresh, contours = motion_det.detect(frame)
            
            display_frame = frame.copy() if args.display else None
            
            class_name = "None"
            conf = 0.0
            
            if motion:
                # Classify
                t0 = time.time()
                # OpenCV uses BGR, we convert to RGB for PIL inside preprocess (wait, predict takes BGR image or PIL?)
                # Find largest contour to crop the insect
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                
                # Add a small 20px padding around the insect
                pad = 20
                y1, y2 = max(0, y - pad), min(frame.shape[0], y + h + pad)
                x1, x2 = max(0, x - pad), min(frame.shape[1], x + w + pad)
                cropped_insect = frame[y1:y2, x1:x2]
                
                from PIL import Image
                rgb_frame = cv2.cvtColor(cropped_insect, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                
                pred = classifier.predict(pil_img)
                inference_latency = time.time() - t0
                
                class_name = pred['class_name']
                conf = pred['confidence']
                
                # Apply CLI threshold override manually
                if conf < args.threshold:
                    class_name = "UNKNOWN"
                    conf = 0.0
                    
                result = temporal_filter.update(class_name, conf)
                state = result['state']
                confirmed_class = result['class_name']
                
                if state == 'CONFIRMED' and confirmed_class:
                    decision = get_decision(confirmed_class, 1.0) # We pass 1.0 since it's already thresholded/confirmed
                    action = decision['action']
                    
                    t1 = time.time()
                    if action == 'CAPTURE':
                        servo.capture()
                    elif action == 'RELEASE':
                        servo.release()
                    actuation_latency = time.time() - t1
                    
                    # Acknowledge actuation to transition to HOLD
                    temporal_filter.acknowledge_actuation()
                    
                    print(f"[{time.strftime('%H:%M:%S')}] ACTUATE: {confirmed_class} -> {action}")

                    
                    if logger:
                        logger.log_event(
                            event_id=str(uuid.uuid4())[:8],
                            predicted_class=confirmed_class,
                            confidence=conf,
                            decision=decision['category'],
                            servo_action=action,
                            inference_latency=inference_latency,
                            actuation_latency=actuation_latency
                        )
                elif state == 'RESET':
                    servo.home()
            else:
                temporal_filter.update("IDLE", 0.0)
                
            if args.display:
                if motion:
                    for c in contours:
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                fps = frames_count / (time.time() - start_time)
                text = f"Class: {class_name} | Conf: {conf:.2f} | FPS: {fps:.1f}"
                cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow("Camera Inference", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    print("Recalibrating background...")
                    motion_det.update_background(frame)
                    
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        
    finally:
        camera.release()
        servo.cleanup()
        if args.display:
            cv2.destroyAllWindows()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
