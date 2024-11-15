import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Set up the video file path
video_path = 'trailer1.mp4'  # Change this to your video file path
cap = cv2.VideoCapture(video_path)

# Check if the video file was successfully opened
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Set the desired frame rate
desired_fps = 30
frame_delay = int(1000 / desired_fps)  # Calculate delay in milliseconds

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.8) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("End of video or failed to capture frame")
            break

        # Convert the image color from BGR to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for face detection
        results = face_detection.process(rgb_frame)

        # Draw face detection bounding boxes
        if results.detections:
            for detection in results.detections:
                print(detection)
                # Draw the detection box using mediapipe drawing utilities
                mp_drawing.draw_detection(frame, detection)

        # Display the frame with face detections
        cv2.imshow('MediaPipe Face Detection', frame)

        # Control the frame rate by adjusting the delay
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
# Release resources
cap.release()
cv2.destroyAllWindows()
