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
                
                face_img = image[y:y+h, x:x+w]
                if face_img.size == 0:
                    continue
                    
                try:
                    emotion = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                    emotion = emotion[0]['dominant_emotion']
                except:
                    emotion = "unknown"
                
                face_results.append({
                    'box': (x, y, x+w, y+h),
                    'emotion': emotion
                })
    
    return face_results

def predict_identity(face_img):
    