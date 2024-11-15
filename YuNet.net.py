import cv2
import face_recognition
import utils
import numpy as np

# Initialize YuNet Face Detector
model_path = 'models/face_detection_yunet_2023mar.onnx'  # Path to your YuNet ONNX model
face_detector = cv2.FaceDetectorYN_create(
    model=model_path,
    config="",
    input_size=(300, 300),
    score_threshold=0.5,
    nms_threshold=0.3,
    top_k=5000
)

# Step 1: Load and Encode Known Faces
known_face_encodings, known_face_names = utils.load_known_faces('data/')

# Set up the video file path
video_path = 'trailer1.mp4'  # Change this to your video file path
cap = cv2.VideoCapture(0)

# Check if the video file was successfully opened
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get original video frame size
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
input_size = (frame_width, frame_height)

# Set the desired frame rate
desired_fps = 30
frame_delay = int(1000 / desired_fps)  # Calculate delay in milliseconds

# Update the YuNet detector's input size to match the video frame size
face_detector.setInputSize(input_size)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("End of video or failed to capture frame")
        break

    # Perform face detection using YuNet
    faces = face_detector.detect(frame)

    # Check if faces are detected
    if faces[1] is not None:
        for face in faces[1]:
            bbox = face[:4].astype(int)
            x, y, w, h = bbox

            # Crop the detected face
            face_image = frame[y:y + h, x:x + w]

            # Convert face image to RGB for face_recognition
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            # Step 4: Encode the Detected Face
            encodings = face_recognition.face_encodings(rgb_face)

            name = "Unknown"
            if encodings:
                encoding = encodings[0]

                # Step 5: Compare with Known Faces
                distances = face_recognition.face_distance(known_face_encodings, encoding)
                best_match_index = np.argmin(distances)

                if distances[best_match_index] < 0.6:  # Set your threshold
                    name = known_face_names[best_match_index]

            # Step 6: Draw Bounding Box and Name
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with face detections
    cv2.imshow('YuNet Face Detection', frame)

    # Control the frame rate by adjusting the delay
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
