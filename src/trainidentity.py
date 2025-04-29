# Train identities for the program to be able recognize who is who
# Input should be like this:
# input_data/
# - person1/
# - - image1.jpg
# - - image2.jpg
# - person2/
# - - image1.jpg
# - - image2.jpg
# - - image3.jpg

import argparse

def main():
    parser = argparse.ArgumentParser(description='Train on datasets of images of people you know are attending an event that are VIP (and should be recognized)')
    parser.add_argument('--input', required=True, help='Input directory containing images of people to train on')
    args = parser.parse_args()

    


if __name__ == "__main__":
    main()
