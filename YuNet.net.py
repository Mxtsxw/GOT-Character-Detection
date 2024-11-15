import json
import os

import cv2
import numpy as np


from process_images import get_face_embeddings, save_face_embeddings, process_images

# Charger les descripteurs stockés dans le fichier JSON
def load_face_embeddings(filename='face_embeddings.json'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {}

model = cv2.dnn.readNetFromONNX('models/arcfaceresnet100-8.onnx')

# Calculer la distance entre deux descripteurs de visages
def compare_embeddings(embedding1, embedding2, threshold=0.6):
    # Calculer la distance euclidienne entre les deux embeddings
    distance = np.linalg.norm(np.array(embedding1) - np.array(embedding2))
    return distance < threshold  # Si la distance est inférieure au seuil, c'est la même personne

# Fonction modifiée pour accepter un tableau NumPy directement
def get_face_embeddings(face_image):
    # Prétraiter l'image du visage si nécessaire (exemple: redimensionnement, normalisation)
    face_image = cv2.resize(face_image, (112, 112))  # Ajuste la taille si nécessaire
    face_image = face_image.astype(np.float32)
    face_image /= 255.0


    # Extraction des embeddings (remplace cette ligne par ta méthode réelle)
    blob = cv2.dnn.blobFromImage(face_image, 1.0, (112, 112), (0, 0, 0), swapRB=True)
    model.setInput(blob)
    embeddings = model.forward()

    return embeddings



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

#stored embeddings
face_embeddings = load_face_embeddings()

# Set up the video file path
video_path = 'trailer1.mp4'
cap = cv2.VideoCapture(video_path)

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
            # Extract bounding box coordinates
            bbox = face[:4].astype(int)
            x, y, w, h = bbox

            # Draw bounding box around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Optionally, draw the confidence score
            confidence_score = face[-1]
            cv2.putText(frame, f'Conf: {confidence_score:.2f}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            #crop the face from the frame
            face_crop = frame[y:y+h, x:x+w]


            #convert the cropped_face image to the format needed for the face embedding
            embeddings = get_face_embeddings(face_crop)


            person_name = 'Unknown'


            # If face embeddings are found, save them and comprare with stored embeddings
            if len(embeddings) > 0:
                embeddings = embeddings.flatten()
                for name, stored_embedding in face_embeddings.items():
                    stored_embedding = np.array(stored_embedding)
                    print(f"Embedding calculé : {embeddings.shape}, Embedding stocké : {stored_embedding.shape}")
                    print(f"Embedding calculé : {embeddings}, Embedding stocké : {stored_embedding}")
                    if compare_embeddings(embeddings[0], stored_embedding):
                        person_name = name
                        break

                # Display the person's name on the frame
                cv2.putText(frame, person_name, (x, y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



    # Display the frame with face detections
    cv2.imshow('YuNet Face Detection', frame)

    # Control the frame rate by adjusting the delay
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

