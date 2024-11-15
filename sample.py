import time

import cv2 as cv
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode
mp_drawing = mp.solutions.drawing_utils





# Create a face detector instance with the live stream mode:
def print_result(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int,frame):
    print('face detector result: {}'.format(result))


options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='models/blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=lambda result, output_image, timestamp_ms: print_result(result, output_image, timestamp_ms, frame)
)

# Assuming you have FaceDetector created with options
with FaceDetector.create_from_options(options) as detector:
    # Use OpenCV’s VideoCapture to start capturing from the webcam.
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Create a loop to read the latest frame from the camera using VideoCapture#read()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # #get current timestamp in milliseconds
        timestamp_ms = int(time.time() * 1000)

        # Process the frame using detect_async
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect_async(mp_image, timestamp_ms)

        # draw the face detection result on the frame
        if detection_result:
            for detection in detection_result.detections:
                mp_drawing.draw_detection(frame, detection)

        # display the resulting frame
        cv.imshow('Face Detection', frame)

        #Break the loop if 'q' is pressed
        if cv.waitKey(1) == ord('q'):
            break

    #when everything is done release the capture (on ferme toutes les fenêtres crées par OpenCv)
    cap.release()
    cv.destroyAllWindows()







