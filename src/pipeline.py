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
import cv2
import numpy as np
import tempfile
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('ultralytics').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

def convert_raw_to_rgb(raw_path):
    """
    Convert RAW image formats (NEF, CR2, ARW, etc.) to a format readable by OpenCV.
    Returns the path to a temporary jpg file.
    """
    try:
        # Try using RawPy for RAW conversion
        import rawpy
        with rawpy.imread(raw_path) as raw:
            rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=False)
            
            # Create a temporary file to save the converted image
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            # Save the image
            cv2.imwrite(temp_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            return temp_path
    except (ImportError, Exception) as e:
        # If RawPy fails or isn't installed, try LibRaw via dcraw
        try:
            import subprocess
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            # Use dcraw to convert the RAW file
            subprocess.run(['dcraw', '-c', '-w', raw_path], stdout=open(temp_path, 'wb'))
            
            # Check if the file was created and has content
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                return temp_path
            else:
                raise Exception("Failed to convert RAW image with dcraw")
        except Exception as dcraw_error:
            raise ValueError(f"Failed to process RAW image {raw_path}: {str(e)}, dcraw error: {str(dcraw_error)}")

class ImageProcessor:
    def __init__(self, input_dir, output_dir, desired_emotion, time_debug=False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.desired_emotion = desired_emotion
        self.temp_files = []  # Track temporary files for cleanup
        self.time_debug = time_debug
        
        # Initialize timing statistics if enabled
        if self.time_debug:
            self.timing_stats = {
                'file_times': {},      # Individual file processing times
                'component_times': {   # Time spent in each component of processing
                    'raw_conversion': 0,
                    'pose_detection': 0,
                    'object_detection': 0,
                    'face_detection': 0,
                    'scoring': 0
                }
            }
        
    def __del__(self):
        # Clean up any temporary files
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
        
    def process_image(self, image_path):
        # Start timing if debug enabled
        if self.time_debug:
            start_time = time.time()
            component_start_time = start_time
        
        image_path = Path(image_path)
        results = {
            'image_name': image_path.name,
            'poses': [],
            'objects': [],
            'faces': []
        }
        
        # Handle RAW formats
        process_path = str(image_path)
        raw_formats = ['.nef', '.raw', '.arw', '.cr2', '.cr3', '.dng', '.orf', '.rw2', '.pef', '.srw']
        if image_path.suffix.lower() in raw_formats:
            try:
                # Time RAW conversion if debug enabled
                if self.time_debug:
                    raw_start_time = time.time()
                
                process_path = convert_raw_to_rgb(str(image_path))
                self.temp_files.append(process_path)
                
                # Record RAW conversion time
                if self.time_debug:
                    self.timing_stats['component_times']['raw_conversion'] += time.time() - raw_start_time
            except Exception as e:
                print(f"Error converting RAW file {image_path}: {str(e)}")
                
                # Record total time for failed RAW conversion
                if self.time_debug:
                    self.timing_stats['file_times'][str(image_path)] = time.time() - start_time
                
                return results
        
        # Process the image with the regular pipeline
        try:
            # Time pose detection
            if self.time_debug:
                component_start_time = time.time()
            
            poses = detect_multiple_poses(process_path)
            for pose in poses:
                results['poses'].append(pose.to_dict('records'))
            
            # Record pose detection time
            if self.time_debug:
                self.timing_stats['component_times']['pose_detection'] += time.time() - component_start_time
                component_start_time = time.time()
            
            # Time object detection
            objects = detect_objects(process_path)
            results['objects'] = objects
            
            # Record object detection time
            if self.time_debug:
                self.timing_stats['component_times']['object_detection'] += time.time() - component_start_time
                component_start_time = time.time()
            
            # Time face detection
            faces = detect_faces(process_path)
            results['faces'] = faces
            
            # Record face detection time
            if self.time_debug:
                self.timing_stats['component_times']['face_detection'] += time.time() - component_start_time
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
        
        # Record total time for this image
        if self.time_debug:
            self.timing_stats['file_times'][str(image_path)] = time.time() - start_time
        
        return results
    
    def score_image(self, results):
        # Start timing for scoring if debug enabled
        if self.time_debug:
            component_start_time = time.time()
        
        emotion_score = 0
        object_score = 0
        face_quality_score = 0
        
        # Process face scores
        if results['faces']:
            for face in results['faces']:
                # Apply face quality to scoring
                face_quality = face.get('face_quality', 1.0)
                
                # Penalize partial faces
                if face.get('is_partial', False):
                    completeness = face.get('face_completeness', 0.5)
                    # Only count partial faces that are mostly visible
                    if completeness >= 0.7:
                        # Reduce score based on how much is cut off
                        quality_factor = completeness
                    else:
                        # Heavily penalize very partial faces
                        quality_factor = 0.3
                else:
                    quality_factor = 1.0
                
                # Calculate emotion match score with quality adjustments
                if face['emotion'].lower() == self.desired_emotion.lower():
                    # Base emotion score weighted by face quality
                    emotion_score += 1 * face_quality * quality_factor
                
                # Add to overall face quality score
                face_quality_score += face_quality * quality_factor
        else:
            # Penalize images with no faces at all
            face_quality_score = -1
        
        # Calculate average face quality if there are faces
        if results['faces']:
            avg_face_quality = face_quality_score / len(results['faces'])
            # Apply a mild bonus for images with multiple good quality faces
            if len(results['faces']) > 1 and avg_face_quality > 0.8:
                face_quality_score *= 1.2
        
        # Process object scores
        for obj in results['objects']:
            for bias in settings.image_raw_bias_settings:
                if obj['label'].lower() == bias['name'].lower():
                    object_score += emotion_score * bias['biasamount']
                    break
        
        # Combine scores - include face quality in the final score
        final_score = emotion_score + object_score
        
        # If we have faces, adjust final score by face quality
        if results['faces'] and face_quality_score > 0:
            # Apply face quality multiplier (neutral at 1.0, can go higher for good faces)
            final_score *= (0.5 + 0.5 * min(face_quality_score, 2.0))
        
        # Store component scores for debugging/analysis
        results['score_components'] = {
            'emotion_score': emotion_score,
            'object_score': object_score,
            'face_quality_score': face_quality_score,
            'final_score': final_score
        }
        
        # Record scoring time
        if self.time_debug:
            self.timing_stats['component_times']['scoring'] += time.time() - component_start_time
        
        return final_score
    
    def process_directory(self):
        all_results = []
        # Include RAW formats in the supported file types
        supported_formats = ['.jpg', '.jpeg', '.png', '.nef', '.raw', '.arw', '.cr2', '.cr3', '.dng', '.orf', '.rw2', '.pef', '.srw']
        image_files = [f for f in self.input_dir.glob('*') if f.suffix.lower() in supported_formats]
        
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