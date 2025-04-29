# The main function of the program, apply the machine learning and return the biases
# Args:
# - Recognition Mode: what modes should be calculated
# - Input Directory: what directory should it read to
# - Output Formatting: how to return the calculations (raw predictions or a rank of files to edit)

# Recognition modes:
# 0 - everything
# 1 - object only
# 2 - pose only
# 3 - identity only
# 4 - object and pose
# 5 - pose and identity
# 6 - identity and object

# Run all of the predictions using all of the models in the program
from recognition.pose import *
from recognition.object import *

class prediction_functions:
    def predict_poses(input_path):
        # Predict poses
        predicted_poses = recognition.pose.detect_multiple_poses()
        return predicted_poses

    def predict_objects():
        pass

class prediction_controller:
    def run_predictions(recognition_mode, input_folder, output_formatting):
        pass

    