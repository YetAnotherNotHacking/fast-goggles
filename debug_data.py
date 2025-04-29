import json
import cv2
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

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
        conf = face.get('confidence', 1.0)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        emotion_text = f"{emotion} {conf:.2f}"
        text_size = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (x1, y2), (x1 + text_size[0], y2 + text_size[1] + 10), color, -1)
        cv2.putText(image, emotion_text, (x1, y2 + text_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def process_summary(summary_path, output_dir=None):
    console = Console()
    summary_path = Path(summary_path)
    
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
            image = cv2.imread(str(image_path))
            if image is None:
                console.print(f"[red]Could not read image: {image_path}[/red]")
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
                    console.print(f"  Face {i}:")
                    console.print(f"    Emotion: {face['emotion']}")
                    console.print(f"    Confidence: {face.get('confidence', 1.0):.2f}")
                    console.print(f"    Box: {face['box']}")
            else:
                console.print("[yellow]No face data found in results[/yellow]")
            
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"debug_{image_path.name}"
                cv2.imwrite(str(output_path), image)
            else:
                window_name = f"Debug: {image_path.name}"
                cv2.imshow(window_name, image)
                cv2.waitKey(0)
                cv2.destroyWindow(window_name)
            
            progress.update(task, advance=1)
    
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