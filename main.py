import argparse
from src.pipeline import ImageProcessor

def main():
    parser = argparse.ArgumentParser(description='Process images for pose, object, and face detection')
    parser.add_argument('--input', required=True, help='Input directory containing images')
    parser.add_argument('--output', required=True, help='Output directory for results')
    
    args = parser.parse_args()
    
    processor = ImageProcessor(args.input, args.output)
    results = processor.process_directory()
    
    print(f"Processed {len(results)} images")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
