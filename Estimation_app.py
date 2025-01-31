import os
import numpy as np
import cv2
from PIL import Image
import streamlit as st

# File paths and constants
DEMO_IMAGE = 'P4-Human-Pose-Estimation\stand.jpg'

BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

inWidth, inHeight = 368, 368

# Load TensorFlow model
net = cv2.dnn.readNetFromTensorflow("C:\Users\shrih\Desktop\Human Pose Estimation\P4-Human-Pose-Estimation\graph_opt.pb")

# Streamlit app setup
st.title("Human Pose Estimation with OpenCV")
st.text('Upload a clear image for pose estimation.')

# File uploader for input image
img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Load and process the image
if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
    if image.shape[-1] == 4:  # Check if the image has 4 channels (RGBA)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)  # Convert RGBA to RGB
elif os.path.exists(DEMO_IMAGE):
    image = np.array(Image.open(DEMO_IMAGE))
    if image.shape[-1] == 4:  # Check if the demo image has 4 channels (RGBA)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)  # Convert RGBA to RGB
else:
    st.error("Demo image not found and no image uploaded. Please upload an image.")
    st.stop()

# Display the original image
st.subheader('Original Image')
st.image(image, caption="Uploaded Image", use_container_width=True)

# Detection threshold slider
thres = st.slider('Detection Threshold', min_value=0, max_value=100, value=20, step=5) / 100

# Pose detection function
def poseDetector(frame):
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()[:, :19, :, :]
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = int(frameWidth * point[0] / out.shape[3])
        y = int(frameHeight * point[1] / out.shape[2])
        points.append((x, y) if conf > thres else None)
    for pair in POSE_PAIRS:
        idFrom, idTo = BODY_PARTS[pair[0]], BODY_PARTS[pair[1]]
        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.circle(frame, points[idFrom], 5, (0, 0, 255), -1)
            cv2.circle(frame, points[idTo], 5, (0, 0, 255), -1)
    return frame

# Perform pose estimation
output = poseDetector(image)

# Display the output image
st.subheader('Pose Estimation Result')
st.image(output, caption="Pose Estimation", use_container_width=True)
