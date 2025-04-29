<p align="center">
  <img src="assets/fastgoggles_main_logo.png" alt="Fast Goggles Logo" width="400"/>
</p>

# Fast Goggles
A utility for proffessional photographers to quickly find photos that best capture the event they are shooting, saving them time so that they are able to spend more of their time on high quality edits for their customers.

## Features:
    ⚡  Fast searching of files and generatoin of custom metadata
    🧠 Makes use of machine learning libraries
    🧰 Easy to integrate with AppleScript or you preffered method of automation
    🔧 CLI interface with Rich TUI for pleasant visual appearance

## Installation
From Source:
```bash
git clone https://github.com/YetAnotherNotHacking/fast-goggles.git
cd fast-goggles
pip3 install -r requirements.txt
# If the above failes, you may want to try (at your own risk):
pip3 install -r requirements.txt --break-system-packages
```
## Usage
First, navigate to the directory that the script is installed into. Then run:
```bash
python3 -m main --input <path> --ouput <path> --desired-emotion <happy,sad,angry,neutral,disgust,etc>
```
Run with --help to see all available options.

## How It Works
Fast Goggles uses basic machine learning to recognize the objects, poses, and emotions in relation to people in the image, it then compares that to the expected emptoin of the event, and based on the amount of a match identified, the score is given. The score may also be increased if certain objects are detected in the image e.g. a large number of people looking the correct direction.

## Documentation
Nothing much currently, though more will come with further development of the program

## Contributing
We welcome contributions of all kinds! Here's how to get started:
*Fork the repository
*Create a new branch (git checkout -b feature/your-feature)
*Commit your changes
*Push your branch (git push origin feature/your-feature)
*Create a Pull Request

Before submitting, please ensure:
    Code passes any lint or test checks
    Features are documented

## License
This project is licensed under the BSD-3-Clause license. Read more about it from the source for more information.