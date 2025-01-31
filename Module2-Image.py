import cv2
import mediapipe as mp

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load an image
image_path = "C:\Users\shrih\Desktop\Human Pose Estimation\P4-Human-Pose-Estimation\HPE.png"
image = cv2.imread(image_path)
if image is None:
    print("Error: Unable to load the image. Check the file path.")
    exit()

# Convert the image to RGB (Mediapipe uses RGB images)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform pose estimation
results = pose.process(image_rgb)

# Check if landmarks are detected
if results.pose_landmarks:
    print("Pose landmarks detected!")

    # Draw landmarks only (no connections)
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        print(f"Landmark {idx}: x={landmark.x}, y={landmark.y}, z={landmark.z}")

        # Get image dimensions
        h, w, _ = image.shape

        # Convert normalized coordinates to pixel coordinates
        cx, cy = int(landmark.x * w), int(landmark.y * h)

        # Draw the keypoints
        cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)  # Green color, filled circle

    # Optional: Draw landmarks and connections on the image
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
    )

    # Display the output images
    cv2.imshow("Pose Landmarks", image)
    cv2.imshow("Pose with Connections", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Release resources
pose.close()


