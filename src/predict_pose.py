# Predict the pose of people from an input image
# Later, prediction models will be used to deteremine what the pose is, and it it's interesting enough to recommend it
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

def detect_multiple_poses(image_path):
    """
    Detects poses of multiple people in an image and returns their landmark coordinates.
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        list: List of pandas DataFrames, where each DataFrame contains pose landmarks 
             for a single person (x, y, z, visibility)
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.1
    )
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    results = pose.process(image_rgb)
    pose_results = []
    if results.pose_landmarks:
        df = pd.DataFrame(columns=['landmark_id', 'x', 'y', 'z', 'visibility'])
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            df.loc[idx] = [idx, landmark.x * width, landmark.y * height, landmark.z, landmark.visibility]
        pose_results.append(df)
    regions = [
        (0, 0, width//2, height//2),           # Top-left
        (width//2, 0, width, height//2),       # Top-right
        (0, height//2, width//2, height),      # Bottom-left
        (width//2, height//2, width, height),  # Bottom-right
        (width//4, height//4, 3*width//4, 3*height//4)  # Center
    ]
    for xmin, ymin, xmax, ymax in regions:
        if xmax - xmin < 100 or ymax - ymin < 100:
            continue
        region_img = image_rgb[ymin:ymax, xmin:xmax]
        with mp_pose.Pose(
            static_image_mode=True, 
            model_complexity=2, 
            min_detection_confidence=0.3
        ) as region_pose:
            region_results = region_pose.process(region_img)
            if region_results.pose_landmarks:
                person_df = pd.DataFrame(columns=['landmark_id', 'x', 'y', 'z', 'visibility'])
                for idx, landmark in enumerate(region_results.pose_landmarks.landmark):
                    x = landmark.x * (xmax - xmin) + xmin
                    y = landmark.y * (ymax - ymin) + ymin
                    person_df.loc[idx] = [idx, x, y, landmark.z, landmark.visibility]
                if not any_similar_pose(person_df, pose_results, threshold=50):
                    pose_results.append(person_df)
    pose.close()
    return pose_results

def any_similar_pose(new_pose_df, existing_poses, threshold=0):
    """
    Checks if a new pose is too similar to any existing pose.
    
    Args:
        new_pose_df (DataFrame): DataFrame containing landmark data for a new pose
        existing_poses (list): List of DataFrames containing landmark data for existing poses
        threshold (float): Pixel distance threshold to consider poses as similar
        
    Returns:
        bool: True if new pose is similar to any existing pose, False otherwise
    """
    if not existing_poses:
        return False
    key_points = [11, 12, 23, 24]  # Shoulder and hip indices
    for existing_df in existing_poses:
        total_dist = 0
        valid_points = 0
        for point in key_points:
            if point in new_pose_df['landmark_id'].values and point in existing_df['landmark_id'].values:
                new_x = new_pose_df.loc[new_pose_df['landmark_id'] == point, 'x'].values[0]
                new_y = new_pose_df.loc[new_pose_df['landmark_id'] == point, 'y'].values[0]
                existing_x = existing_df.loc[existing_df['landmark_id'] == point, 'x'].values[0]
                existing_y = existing_df.loc[existing_df['landmark_id'] == point, 'y'].values[0]
                dist = np.sqrt((new_x - existing_x)**2 + (new_y - existing_y)**2)
                total_dist += dist
                valid_points += 1
        if valid_points > 0:
            avg_dist = total_dist / valid_points
            if avg_dist < threshold:
                return True
    return False
def visualize_poses(image_path, pose_results):
    """
    Visualizes detected poses on the input image.
    
    Args:
        image_path (str): Path to the input image
        pose_results (list): List of DataFrames containing landmark data
        
    Returns:
        numpy.ndarray: Image with poses visualized
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32)
    ]
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),    # Maroon
        (0, 128, 0),    # Dark Green
        (0, 0, 128),    # Navy
        (128, 128, 0)   # Olive
    ]
    for i, df in enumerate(pose_results):
        color = colors[i % len(colors)]
        for _, row in df.iterrows():
            x, y = int(row['x']), int(row['y'])
            cv2.circle(image, (x, y), 5, color, -1)
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx in df['landmark_id'].values and end_idx in df['landmark_id'].values:
                start_x = int(df.loc[df['landmark_id'] == start_idx, 'x'].values[0])
                start_y = int(df.loc[df['landmark_id'] == start_idx, 'y'].values[0])
                end_x = int(df.loc[df['landmark_id'] == end_idx, 'x'].values[0])
                end_y = int(df.loc[df['landmark_id'] == end_idx, 'y'].values[0])
                cv2.line(image, (start_x, start_y), (end_x, end_y), color, 2)
    return image
# Test - It works well enough, though really it could be better for more effecient function of the program
# if __name__ == "__main__":
#     # ONLY UNCOMMENT THIS IF YOU ARE TESTING, RECOMMENT THE ENTIRE IF__NAME__ == MAIN LOGIC TO DISABLE
#     image_path = "testdata/manypeople.jpg"
#     poses = detect_multiple_poses(image_path)
#     print(f"Detected {len(poses)} people in the image")
#     for i, pose_df in enumerate(poses):
#         print(f"\nPerson {i+1} landmarks:")
#         print(pose_df)
#     result_image = visualize_poses(image_path, poses)
#     # Save or display the result
#     # cv2.imwrite("poses_visualization.jpg", result_image)
#     # Uncomment below to display (if running in an environment with GUI)
#     cv2.imshow("Poses", result_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()