#!/usr/bin/env python3
"""
Test script for the --process-time-debug flag.
This creates a few test images and runs the pipeline with timing enabled.
"""

import os
import sys
import tempfile
import shutil
import cv2
import numpy as np
from pathlib import Path
import subprocess
import argparse

def create_test_images(output_dir, count=5):
    """Create some test images for processing."""
    # Create simple test images with varying content
    for i in range(count):
        # Create a base image (random color)
        color = np.random.randint(0, 255, 3).tolist()
        # Create the image with the correct data type
        img = np.ones((500, 500, 3), dtype=np.uint8)
        img[:, :, 0] = color[0]
        img[:, :, 1] = color[1]
        img[:, :, 2] = color[2]
        
        # Add a simple face-like circle (to trigger face detection)
        center = (250, 250)
        radius = 100
        cv2.circle(img, center, radius, (255, 200, 200), -1)  # Face
        
        # Add eyes
        cv2.circle(img, (200, 200), 20, (255, 255, 255), -1)   # Left eye
        cv2.circle(img, (300, 200), 20, (255, 255, 255), -1)   # Right eye
        
        # Add mouth
        cv2.rectangle(img, (225, 300), (275, 320), (150, 100, 100), -1)  # Mouth
        
        # Add text to the image
        cv2.putText(img, f"Test {i+1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Save the image
        filename = f"test_image_{i+1}.jpg"
        file_path = os.path.join(output_dir, filename)
        cv2.imwrite(file_path, img)
        print(f"Created test image: {file_path}")
        
    print(f"Created {count} test images in {output_dir}")
    return count

def main():
    parser = argparse.ArgumentParser(description='Test the process-time-debug flag')
    parser.add_argument('--count', type=int, default=5, help='Number of test images to create')
    parser.add_argument('--emotion', default='happy', help='Emotion to search for')
    
    args = parser.parse_args()
    
    # Create temporary directories for input and output
    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        # Create test images
        count = create_test_images(input_dir, args.count)
        
        # Run the main script with timing enabled
        cmd = [
            "python3", "-m", "main",
            "--input", input_dir,
            "--output", output_dir,
            "--desired-emotion", args.emotion,
            "--process-time-debug"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        
        print(f"\nProcess completed with return code: {result.returncode}")

if __name__ == "__main__":
    main() 