import mediapipe as mp
import cv2

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection = 0, min_detection_confidence = 0.5) as face_detection:
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break

        temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(temp_frame)
        print(results.detections)

        # if results.detections:
        #     for detection in results.detections:
        #         mp_drawing.draw_detection(frame, detection)

        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()