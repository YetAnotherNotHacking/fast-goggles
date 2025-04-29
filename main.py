import argparse
import logging
import os
import warnings
from src.pipeline import ImageProcessor
from src import settings
from rich.console import Console
from rich.table import Table

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('ultralytics').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

def main():
    supported_emotions = "angry, disgust, fear, happy, sad, surprise, neutral"
    parser = argparse.ArgumentParser(description='Process images for pose, object, and face detection')
    parser.add_argument('--input', required=True, help='Input directory containing images')
    parser.add_argument('--output', required=True, help='Output directory for results')
    parser.add_argument('--desired-emotion', required=True, help=f'Target emotion to score images by. Supported emotions: {supported_emotions}')
    
    args = parser.parse_args()
    
    processor = ImageProcessor(args.input, args.output, args.desired_emotion)
    results = processor.process_directory()
    
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="dim")
    table.add_column("Image")
    table.add_column("Score")
    table.add_column("Faces with Emotion")
    table.add_column("Relevant Objects")
    
    for i, result in enumerate(results, 1):
        emotion_faces = sum(1 for face in result['faces'] if face['emotion'].lower() == args.desired_emotion.lower())
        relevant_objects = [obj['label'] for obj in result['objects'] if obj['label'] in settings.OBJECT_BIASES]
        
        table.add_row(
            str(i),
            result['image_name'],
            f"{result['score']:.2f}",
            str(emotion_faces),
            ", ".join(relevant_objects) if relevant_objects else "None"
        )
    
    console.print("\n[bold]Top Images Ranked by Emotion Match:[/bold]")
    console.print(table)

if __name__ == "__main__":
    main()
