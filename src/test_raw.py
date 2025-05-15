#!/usr/bin/env python3
"""
Test script for RAW image support in Fast Goggles.
This script tests the RAW conversion functionality separately from the main pipeline.
"""

import os
import sys
import argparse
import cv2
from pathlib import Path
import tempfile

def convert_raw_to_rgb(raw_path):
    """
    Convert RAW image formats (NEF, CR2, ARW, etc.) to a format readable by OpenCV.
    Returns the path to a temporary jpg file.
    """
    # Get the file extension and make it lowercase
    ext = Path(raw_path).suffix.lower()
    
    # Check if the file exists
    if not os.path.exists(raw_path):
        print(f"Error: File {raw_path} does not exist")
        return None
    
    # First, try rawpy (preferred method)
    try:
        import rawpy
        print(f"Using rawpy to convert {raw_path}")
        
        with rawpy.imread(raw_path) as raw:
            rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=False)
            
            # Create a temporary file to save the converted image
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            # Save the image
            cv2.imwrite(temp_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            print(f"Successfully converted with rawpy to {temp_path}")
            return temp_path
    except ImportError:
        print("rawpy is not installed. Please install it with: pip install rawpy")
    except Exception as e:
        print(f"rawpy error: {str(e)}")
    
    # If rawpy fails, try dcraw
    try:
        import subprocess
        print(f"Trying dcraw for {raw_path}")
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Use dcraw to convert the RAW file
        subprocess.run(['dcraw', '-c', '-w', raw_path], stdout=open(temp_path, 'wb'))
        
        # Check if the file was created and has content
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            print(f"Successfully converted with dcraw to {temp_path}")
            return temp_path
        else:
            print("dcraw produced an empty file")
    except Exception as e:
        print(f"dcraw error: {str(e)}")
    
    # Last resort: try to open directly with OpenCV
    # This will likely only work for some formats or with additional OpenCV plugins
    print(f"Trying direct OpenCV reading for {raw_path}")
    img = cv2.imread(raw_path)
    if img is not None and img.size > 0:
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        cv2.imwrite(temp_path, img)
        print(f"Successfully read directly with OpenCV and saved to {temp_path}")
        return temp_path
    else:
        print("Direct OpenCV reading failed")
    
    print(f"All conversion methods failed for {raw_path}")
    return None

def display_image(image_path):
    """Display the image using OpenCV"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image {image_path}")
        return False
    
    print(f"Image dimensions: {img.shape}")
    
    # Resize if the image is too large
    max_height = 800
    if img.shape[0] > max_height:
        scale = max_height / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1] * scale), max_height))
    
    window_name = "RAW Image Test"
    cv2.imshow(window_name, img)
    print("Press any key to close the image window")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    return True

def main():
    parser = argparse.ArgumentParser(description='Test RAW image conversion')
    parser.add_argument('file', help='Path to a RAW image file')
    parser.add_argument('--save', help='Path to save the converted image (optional)')
    
    args = parser.parse_args()
    
    print(f"Testing RAW conversion for {args.file}")
    converted_path = convert_raw_to_rgb(args.file)
    
    if converted_path:
        if args.save:
            import shutil
            shutil.copy(converted_path, args.save)
            print(f"Saved converted image to {args.save}")
        
        # Display the converted image
        display_image(converted_path)
        
        # Clean up the temporary file
        try:
            os.unlink(converted_path)
        except:
            pass
        
        return 0
    else:
        print("Conversion failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 