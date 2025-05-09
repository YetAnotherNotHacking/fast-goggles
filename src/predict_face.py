import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from deepface import DeepFace

def detect_faces(image_path):
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    
    face_results = []
    
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as face_detection:
        results = face_detection.process(image_rgb)
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                # Calculate face quality metrics
                face_quality = 1.0
                face_completeness = 1.0
                is_partial = False
                
                # Check if face is cut off at image boundaries
                if x < 0 or y < 0 or x + w > width or y + h > height:
                    is_partial = True
                    # Calculate how much of the face is visible (0.0-1.0)
                    visible_x = max(0, min(width, x + w)) - max(0, x)
                    visible_y = max(0, min(height, y + h)) - max(0, y)
                    visible_area = visible_x * visible_y
                    total_area = w * h
                    face_completeness = visible_area / total_area if total_area > 0 else 0
                    # Penalize cut-off faces
                    face_quality *= face_completeness
                
                # Adjust coordinates to be within image boundaries
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)
                
                # Skip faces that are too small or barely visible
                if w < 20 or h < 20 or face_completeness < 0.5:
                    continue
                
                face_img = image[y:y+h, x:x+w]
                if face_img.size == 0:
                    continue
                    
                try:
                    emotion = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                    emotion = emotion[0]['dominant_emotion']
                except:
                    emotion = "unknown"
                
                # Calculate face size relative to image (0.0-1.0)
                face_size_ratio = (w * h) / (width * height)
                
                # Adjust quality based on face size
                # Penalize very small faces
                if face_size_ratio < 0.01:
                    face_quality *= 0.5
                # Slightly boost medium-sized faces that are the focus
                elif 0.05 <= face_size_ratio <= 0.3:
                    face_quality *= 1.2
                
                face_results.append({
                    'box': (x, y, x+w, y+h),
                    'emotion': emotion,
                    'is_partial': is_partial,
                    'face_completeness': face_completeness,
                    'face_quality': min(face_quality, 1.0),  # Cap at 1.0
                    'face_size_ratio': face_size_ratio
                })
    
    return face_results

def predict_identity(face_img):
    pass
    