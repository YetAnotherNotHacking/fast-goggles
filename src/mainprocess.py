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
from predict_pose import *
from predict_object import detect_objects
from settings import *

# Format all of the settings in an easy to manage layout for these functions
threshhold_object = object_confidence_threshhold
threshhold_pose = pose_visibility_threshhold

class predict_data_from_image:
    def predict_present_objects(input_image_path):
        present_objects = []
        predicted_objects = detect_objects(input_image_path)
        for found_object in predicted_objects:
            if found_object['confidence'] >= threshhold_object:
                present_objects.append(found_object)
            elif found_object['confidence'] < threshhold_object:
                print("DEBUG - too low confidence!")
        return present_objects
    def predict_present_poses(input_image_path):
        present_poses = []
        predicted_poses = detect_multiple_poses(input_image_path)
        for pose_df in predicted_poses:
            visible_landmarks = []
            for landmark in pose_df.to_dict(orient='records'):
                if landmark['visibility'] >= threshhold_pose:
                    visible_landmarks.append(landmark)
                elif landmark['visibility'] < threshhold_pose:
                    print("DEBUG - Pose not visible enough to add.")
            if visible_landmarks:
                present_poses.append(visible_landmarks)
        return present_poses


# Test all of the outputs:
if __name__ == "__main__":
    test_input_file_objects = 'testdata/testobject.png'
    test_output_objects = predict_data_from_image.predict_present_objects(test_input_file_objects)            
    print(f"Test output objects {test_output_objects}")

    test_input_file_pose = 'testdata/manypeople.jpg'
    test_output_poses = predict_data_from_image.predict_present_poses(test_input_file_pose)
    print(f"Test output poses {test_output_poses}")