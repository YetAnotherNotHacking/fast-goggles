# Run all of the predictions using all of the models in the program
from recognition.pose import *
from recognition.object import *
import rich

class predictionhandler:
    def predict_poses(input_path):
        # Predict poses
        predicted_poses = recognition.pose.detect_multiple_poses()
        return predicted_poses

    def predict_objects():
        pass
