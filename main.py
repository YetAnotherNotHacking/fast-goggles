import argparse
import logging
import os
import warnings
import time
import statistics
from pathlib import Path
from src.pipeline import ImageProcessor
from src import settings
from rich.console import Console
from rich.table import Table
from rich import box

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
    supported_formats = ".jpg, .jpeg, .png, .nef, .raw, .arw, .cr2, .cr3, .dng, .orf, .rw2, .pef, .srw"
    
    parser = argparse.ArgumentParser(description='Process images for pose, object, and face detection')
    parser.add_argument('--input', required=True, 
                        help=f'Input directory containing images (supported formats: {supported_formats})')
    parser.add_argument('--output', required=True, 
                        help='Output directory for results')
    parser.add_argument('--desired-emotion', required=True, 
                        help=f'Target emotion to score images by. Supported emotions: {supported_emotions}')
    parser.add_argument('--process-time-debug', action='store_true',
                        help='Display detailed processing time statistics')
    
    args = parser.parse_args()
    
    console = Console()
    
    # Track overall processing time
    start_time = time.time()
    
    # Initialize the processor
    processor = ImageProcessor(args.input, args.output, args.desired_emotion, 
                              time_debug=args.process_time_debug)
    
    # Process the directory and get the results
    results = processor.process_directory()
    
    # Calculate total processing time
    total_time = time.time() - start_time
    
    # Display results table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="dim")
    table.add_column("Image")
    table.add_column("Score")
    table.add_column("Faces with Emotion")
    table.add_column("Face Quality")
    table.add_column("Relevant Objects")
    
    for i, result in enumerate(results[:20], 1):
        face_info = []
        face_quality_info = []
        
        for face in result['faces']:
            emotion_str = face['emotion']
            if emotion_str.lower() == args.desired_emotion.lower():
                emotion_str = f"[green]{emotion_str}[/green]"
            face_info.append(emotion_str)
            
            # Format face quality information
            quality = face.get('face_quality', 1.0)
            is_partial = face.get('is_partial', False)
            completeness = face.get('face_completeness', 1.0)
            
            quality_color = "green"
            if is_partial:
                if completeness < 0.7:
                    quality_color = "red"
                else:
                    quality_color = "yellow"
            elif quality < 0.7:
                quality_color = "yellow"
                
            quality_str = f"[{quality_color}]{quality:.2f}"
            if is_partial:
                quality_str += f" (partial: {completeness:.2f})[/{quality_color}]"
            else:
                quality_str += f"[/{quality_color}]"
                
            face_quality_info.append(quality_str)
        
        object_info = [
            obj['label'] for obj in result['objects'] 
            if any(obj['label'].lower() == bias['name'].lower() for bias in settings.image_raw_bias_settings)
        ]
        
        table.add_row(
            str(i),
            result['image_name'],
            f"{result['score']:.2f}",
            ", ".join(face_info) if face_info else "No faces",
            ", ".join(face_quality_info) if face_quality_info else "N/A",
            ", ".join(object_info) if object_info else "No relevant objects"
        )
    
    console.print(table)
    console.print(f"\nResults saved to: {args.output}")
    
    # Display timing information if requested
    if args.process_time_debug and hasattr(processor, 'timing_stats'):
        stats = processor.timing_stats
        
        if stats['file_times']:
            # Calculate statistics
            avg_time = statistics.mean(stats['file_times'].values())
            max_time_file = max(stats['file_times'].items(), key=lambda x: x[1])
            min_time_file = min(stats['file_times'].items(), key=lambda x: x[1])
            raw_file_count = len([f for f in stats['file_times'] if Path(f).suffix.lower() in 
                                ['.nef', '.raw', '.arw', '.cr2', '.cr3', '.dng', '.orf', '.rw2', '.pef', '.srw']])
            
            # Create timing table
            timing_table = Table(title="Processing Time Statistics", box=box.ROUNDED)
            timing_table.add_column("Metric", style="cyan")
            timing_table.add_column("Value", style="green")
            
            timing_table.add_row("Total processing time", f"{total_time:.2f} seconds")
            timing_table.add_row("Total files processed", str(len(stats['file_times'])))
            timing_table.add_row("Regular image files", str(len(stats['file_times']) - raw_file_count))
            timing_table.add_row("RAW image files", str(raw_file_count))
            timing_table.add_row("Average time per file", f"{avg_time:.2f} seconds")
            timing_table.add_row("Fastest file", f"{min_time_file[0]} ({min_time_file[1]:.2f} seconds)")
            timing_table.add_row("Slowest file", f"{max_time_file[0]} ({max_time_file[1]:.2f} seconds)")
            
            # Add component timing if available
            if stats['component_times']:
                timing_table.add_section()
                for component, time_taken in stats['component_times'].items():
                    avg_component_time = time_taken / len(stats['file_times']) if len(stats['file_times']) > 0 else 0
                    timing_table.add_row(
                        f"Average {component} time", 
                        f"{avg_component_time:.2f} seconds"
                    )
            
            console.print("\n")
            console.print(timing_table)
            
            # Print individual file timing if there are many files
            if len(stats['file_times']) > 1:
                file_table = Table(title="Individual File Processing Times", box=box.ROUNDED)
                file_table.add_column("File", style="cyan")
                file_table.add_column("Time (seconds)", style="green")
                file_table.add_column("Type", style="yellow")
                
                # Sort by processing time (descending)
                sorted_files = sorted(stats['file_times'].items(), key=lambda x: x[1], reverse=True)
                
                for file_path, process_time in sorted_files:
                    file_type = "RAW" if Path(file_path).suffix.lower() in [
                        '.nef', '.raw', '.arw', '.cr2', '.cr3', '.dng', '.orf', '.rw2', '.pef', '.srw'
                    ] else "Regular"
                    file_table.add_row(str(file_path), f"{process_time:.2f}", file_type)
                
                console.print("\n")
                console.print(file_table)

if __name__ == "__main__":
    main()
