import json
import cv2
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import os
import tempfile

# Import the RAW conversion function from pipeline
try:
    from src.pipeline import convert_raw_to_rgb
except ImportError:
    # Fallback implementation if imported from a different context
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

def draw_pose(image, pose_data, color=(0, 255, 0)):
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32)
    ]
    
    for landmark in pose_data:
        x, y = int(landmark['x']), int(landmark['y'])
        cv2.circle(image, (x, y), 5, color, -1)
    
    for connection in connections:
        start_idx, end_idx = connection
        start_landmark = next((l for l in pose_data if l['landmark_id'] == start_idx), None)
        end_landmark = next((l for l in pose_data if l['landmark_id'] == end_idx), None)
        
        if start_landmark and end_landmark:
            start_x, start_y = int(start_landmark['x']), int(start_landmark['y'])
            end_x, end_y = int(end_landmark['x']), int(end_landmark['y'])
            cv2.line(image, (start_x, start_y), (end_x, end_y), color, 2)

def draw_objects(image, objects, color=(255, 0, 0)):
    for obj in objects:
        x1, y1, x2, y2 = obj['box']
        label = obj['label']
        conf = obj['confidence']
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def draw_faces(image, faces, color=(0, 0, 255)):
    for face in faces:
        x1, y1, x2, y2 = face['box']
        emotion = face['emotion']
        
        # Determine the color based on quality
        is_partial = face.get('is_partial', False)
        face_quality = face.get('face_quality', 1.0)
        face_completeness = face.get('face_completeness', 1.0)
        
        if is_partial and face_completeness < 0.7:
            # Red for badly partial faces
            box_color = (0, 0, 255)  # Red
        elif is_partial or face_quality < 0.7:
            # Yellow for partial but mostly visible faces or low quality
            box_color = (0, 255, 255)  # Yellow
        else:
            # Green for good quality faces
            box_color = (0, 255, 0)  # Green
        
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
        
        # Draw face quality info
        quality_text = f"{emotion}"
        if is_partial:
            quality_text += f" Q:{face_quality:.2f} P:{face_completeness:.2f}"
        else:
            quality_text += f" Q:{face_quality:.2f}"
            
        text_size = cv2.getTextSize(quality_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (x1, y2), (x1 + text_size[0], y2 + text_size[1] + 10), box_color, -1)
        cv2.putText(image, quality_text, (x1, y2 + text_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def process_summary(summary_path, output_dir=None):
    console = Console()
    summary_path = Path(summary_path)
    temp_files = []  # Track temporary files for cleanup
    
    try:
        with open(summary_path) as f:
            results = json.load(f)
        
        if isinstance(results, dict):
            results = [results]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Processing images...", total=len(results))
            
            for result in results:
                image_path = summary_path.parent / result['image_name']
                
                # Handle RAW formats
                raw_formats = ['.nef', '.raw', '.arw', '.cr2', '.cr3', '.dng', '.orf', '.rw2', '.pef', '.srw']
                if image_path.suffix.lower() in raw_formats:
                    try:
                        console.print(f"[yellow]Converting RAW image: {image_path}[/yellow]")
                        converted_path = convert_raw_to_rgb(str(image_path))
                        temp_files.append(converted_path)
                        image = cv2.imread(converted_path)
                    except Exception as e:
                        console.print(f"[red]Error converting RAW image {image_path}: {str(e)}[/red]")
                        progress.update(task, advance=1)
                        continue
                else:
                    image = cv2.imread(str(image_path))
                
                if image is None:
                    console.print(f"[red]Could not read image: {image_path}[/red]")
                    progress.update(task, advance=1)
                    continue
                
                console.print(f"\n[cyan]Processing {image_path.name}[/cyan]")
                console.print(f"Available keys: {list(result.keys())}")
                
                for pose in result['poses']:
                    draw_pose(image, pose)
                
                draw_objects(image, result['objects'])
                
                faces = result.get('faces', [])
                if faces:
                    draw_faces(image, faces)
                    console.print(f"\n[cyan]Faces detected in {image_path.name}:[/cyan]")
                    for i, face in enumerate(faces, 1):
                        is_partial = face.get('is_partial', False)
                        face_quality = face.get('face_quality', 1.0)
                        
                        quality_color = "green"
                        if is_partial:
                            completeness = face.get('face_completeness', 1.0)
                            if completeness < 0.7:
                                quality_color = "red"
                            else:
                                quality_color = "yellow"
                        elif face_quality < 0.7:
                            quality_color = "yellow"
                        
                        console.print(f"  Face {i}:")
                        console.print(f"    Emotion: {face['emotion']}")
                        console.print(f"    Quality: [{quality_color}]{face_quality:.2f}[/{quality_color}]")
                        if is_partial:
                            console.print(f"    Partial: [{quality_color}]Yes (completeness: {face.get('face_completeness', 1.0):.2f})[/{quality_color}]")
                        console.print(f"    Box: {face['box']}")
                else:
                    console.print("[yellow]No face data found in results[/yellow]")
                
                # Display score components if available
                if 'score_components' in result:
                    console.print(f"\n[cyan]Score breakdown for {image_path.name}:[/cyan]")
                    for comp_name, comp_value in result['score_components'].items():
                        console.print(f"  {comp_name}: {comp_value:.2f}")
                
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"debug_{image_path.stem}.jpg"
                    cv2.imwrite(str(output_path), image)
                else:
                    window_name = f"Debug: {image_path.name}"
                    cv2.imshow(window_name, image)
                    cv2.waitKey(0)
                    cv2.destroyWindow(window_name)
                
                progress.update(task, advance=1)
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
    
    if output_dir:
        console.print(f"\n[green]Debug images saved to: {output_dir}[/green]")
    else:
        console.print("\n[green]Debug visualization complete. Press any key to close windows.[/green]")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate debug images with detected elements')
    parser.add_argument('--summary', required=True, help='Path to summary.json')
    parser.add_argument('--output', help='Output directory for debug images (optional)')
    
    args = parser.parse_args()
    process_summary(args.summary, args.output) 