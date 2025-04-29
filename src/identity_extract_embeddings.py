import os
import cv2
from imutils import paths
import numpy as np
import imutils
import pickle

BASE_DIR = os.path.dirname(__file__)
print("[INFO] BASE DIR: ", BASE_DIR)
print("[INFO] Loading face detector...")
protoPath = os.path.join(BASE_DIR, "face_detection_model/deploy.prototxt")
modelPath = os.path.join(BASE_DIR, "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")
embedding_model = os.path.join(BASE_DIR, 'openface_nn4.small2.v1.t7')
dataset = os.path.join(BASE_DIR, 'dataset')
embeddings = os.path.join(BASE_DIR, 'output/embeddings.pickle')
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
print("[INFO] Loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embedding_model)
print("[INFO] Load image dataset..")
imagePaths = list(paths.list_images(dataset))
print("[DEBUG] Image Paths: ", imagePaths)
knownEmbeddings = []
knownNames = []
total = 0