<p align="center">
  <img src="assets/fastgoggles_main_logo.png" alt="Fast Goggles Logo" width="150"/>
</p>

# Fast Goggles
A utility for proffessional photographers to quickly find photos that best capture the event they are shooting, saving them time so that they are able to spend more of their time on high quality edits for their customers.

## Features:
 - ⚡  Fast searching of files and generatoin of custom metadata
 - 🧠 Makes use of machine learning libraries
 - 🧰 Easy to integrate with AppleScript or you preffered method of automation
 - 🔧 CLI interface with Rich TUI for pleasant visual appearance
 - 📸 Support for both JPEG/PNG and RAW image formats (NEF, RAW, ARW, CR2, etc.)
 - ⏱️ Performance analysis with detailed timing statistics

## Installation
From Source:
```bash
git clone https://github.com/YetAnotherNotHacking/fast-goggles.git
cd fast-goggles
pip3 install -r requirements.txt
# If the above failes, you may want to try (at your own risk):
pip3 install -r requirements.txt --break-system-packages
```

### RAW Image Support
For RAW image processing, Fast Goggles uses the rawpy library. In some cases, you might also need to install the libraw development package for your system:

**Ubuntu/Debian:**
```bash
sudo apt-get install libraw-dev
```

**macOS (with Homebrew):**
```bash
brew install libraw
```

**Windows:**
The rawpy package should include the necessary libraries.

## Usage
First, navigate to the directory that the script is installed into. Then run:
```bash
python3 -m main --input <path> --output <path> --desired-emotion <happy,sad,angry,neutral,disgust,etc>
```
Run with `--help` to see all available options.

### Command Line Options

```
--input INPUT         Input directory containing images
--output OUTPUT       Output directory for results
--desired-emotion DESIRED_EMOTION
                      Target emotion to score images by
--process-time-debug  Display detailed processing time statistics
```

### Performance Analysis

To analyze processing performance, use the `--process-time-debug` flag:

```bash
python3 -m main --input <path> --output <path> --desired-emotion happy --process-time-debug
```

This will generate detailed timing statistics after processing completes, including:
- Total processing time
- Average time per file
- Fastest and slowest files
- Time spent in different processing components (face detection, object detection, etc.)
- Separate statistics for RAW image files vs. regular image files

This is particularly useful for:
- Debugging slow performance issues
- Understanding the impact of RAW image processing
- Optimizing batch processing of large image collections

### Supported Image Formats
Fast Goggles supports the following image formats:
- JPEG/JPG
- PNG
- NEF (Nikon RAW)
- RAW (Generic RAW)
- ARW (Sony RAW)
- CR2/CR3 (Canon RAW)
- DNG (Adobe Digital Negative)
- ORF (Olympus RAW)
- RW2 (Panasonic RAW)
- PEF (Pentax RAW)
- SRW (Samsung RAW)

## How It Works
Fast Goggles uses basic machine learning to recognize the objects, poses, and emotions in relation to people in the image, it then compares that to the expected emptoin of the event, and based on the amount of a match identified, the score is given. The score may also be increased if certain objects are detected in the image e.g. a large number of people looking the correct direction.

The tool also evaluates the quality of faces in images, ensuring that partial or poorly framed faces are given lower scores. Images with complete, well-positioned faces will be prioritized in recommendations.

## Documentation
Nothing much currently, though more will come with further development of the program

## Contributing
We welcome contributions of all kinds! Here's how to get started:
 - Fork the repository
 - Create a new branch (git checkout -b feature/your-feature)
 - Commit your changes
 - Push your branch (git push origin feature/your-feature)
 - Create a Pull Request

Before submitting, please ensure:
 - Code passes any lint or test checks
 - Features are documented

## License
This project is licensed under the BSD-3-Clause license. Read more about it from the source for more information.