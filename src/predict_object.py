def detect_objects(image_path):
    """
    Detect objects in an image and return their coordinates and labels.
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        list: List of dictionaries, each containing:
              - 'label': The detected object class name
              - 'confidence': Detection confidence score
              - 'box': Bounding box coordinates (x1, y1, x2, y2)
    """
    try:
        from ultralytics import YOLO
        import cv2
        model = YOLO('yolov8n.pt')  # n (nano) for speed, you can use 's', 'm', 'l', or 'x' for better accuracy
        results = model(image_path)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = r.names[class_id]
                detections.append({
                    'label': class_name,
                    'confidence': confidence,
                    'box': (x1, y1, x2, y2)
                })
        return detections
    except Exception as e:
        print(f"Error detecting objects: {str(e)}")
        return []
# Test:
# if __name__ == "__main__":
#     image_path = "testdata/testobject.png"
#     detected_objects = detect_objects(image_path)
#     print(f"Found {len(detected_objects)} objects:")
#     for obj in detected_objects:
#         print(f"- {obj['label']} (confidence: {obj['confidence']:.2f}) at {obj['box']}")
#     import cv2
#     import numpy as np
#     image = cv2.imread(image_path)
#     for obj in detected_objects:
#         x1, y1, x2, y2 = obj['box']
#         label = obj['label']
#         conf = obj['confidence']
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         text = f"{label} {conf:.2f}"
#         text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
#         cv2.rectangle(image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 255, 0), -1)
#         cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
#     output_path = "output_detection.jpg"
#     cv2.imwrite(output_path, image)
#     print(f"Annotated image saved to {output_path}")