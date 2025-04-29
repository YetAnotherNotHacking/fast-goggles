import os
import json
from pathlib import Path
import pandas as pd
import warnings
from .predict_pose import detect_multiple_poses
from .predict_object import detect_objects
from .predict_face import detect_faces
from . import settings
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('ultralytics').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

class ImageProcessor:
    def __init__(self, input_dir, output_dir, desired_emotion):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.desired_emotion = desired_emotion
        
    def process_image(self, image_path):
        image_path = Path(image_path)
        results = {
            'image_name': image_path.name,
            'poses': [],
            'objects': [],
            'faces': []
        }
        
        poses = detect_multiple_poses(str(image_path))
        for pose in poses:
            results['poses'].append(pose.to_dict('records'))
            
        objects = detect_objects(str(image_path))
        results['objects'] = objects
        
        faces = detect_faces(str(image_path))
        results['faces'] = faces
        
        return results
    
    def score_image(self, results):
        emotion_score = 0
        object_score = 0
        
        for face in results['faces']:
            if face['emotion'].lower() == self.desired_emotion.lower():
                emotion_score += 1
        
        for obj in results['objects']:
            for bias in settings.image_raw_bias_settings:
                if obj['label'].lower() == bias['name'].lower():
                    object_score += emotion_score * bias['biasamount']
                    break
        
        return emotion_score + object_score
    
    def process_directory(self):
        all_results = []
        image_files = [f for f in self.input_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Processing images...", total=len(image_files))
            
            for image_path in image_files:
                try:
                    results = self.process_image(image_path)
                    results['score'] = self.score_image(results)
                    all_results.append(results)
                    
                    output_path = self.output_dir / f"{image_path.stem}_results.json"
                    with open(output_path, 'w') as f:
                        json.dump(results, f, indent=2)
                        
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                
                progress.update(task, advance=1)
        
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        return sorted(all_results, key=lambda x: x['score'], reverse=True) 