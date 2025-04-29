import os
import json
from pathlib import Path
import pandas as pd
from predict_pose import detect_multiple_poses
from predict_object import detect_objects
from predict_face import detect_faces

class ImageProcessor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    def process_directory(self):
        all_results = []
        for image_path in self.input_dir.glob('*'):
            if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    results = self.process_image(image_path)
                    all_results.append(results)
                    
                    output_path = self.output_dir / f"{image_path.stem}_results.json"
                    with open(output_path, 'w') as f:
                        json.dump(results, f, indent=2)
                        
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
        
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        return all_results 