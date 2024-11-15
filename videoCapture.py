import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Set up the webcam and Face Detection
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
 
        if not ret:
            print("Failed to capture image")
            break

        # Convert the image color from BGR to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for face detection
        results = face_detection.process(rgb_frame)

        # Draw face detection bounding boxes
        if results.detections:
            for detection in results.detections:
                # Draw the detection box using mediapipe drawing utilities
                mp_drawing.draw_detection(frame, detection)

        # Display the frame with face detections
        cv2.imshow('MediaPipe Face Detection', frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
