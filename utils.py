import os
import pickle
import face_recognition
import cv2
import numpy as np


def detect_faces_mediapipe(image, face_detection):
    """
    Function to perform face detection with MediaPipe
    """

    # Convert image to RGB for MediaPipe processing
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image
    results = face_detection.process(rgb_image)

    # Draw bounding boxes
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image


def detect_faces_yunet(image, face_detector):
    """
    Function to perform face detection with YuNet
    """

    # Update YuNet input size to match image
    h, w, _ = image.shape
    face_detector.setInputSize((w, h))

    # Perform detection
    faces = face_detector.detect(image)

    # Draw bounding boxes
    if faces[1] is not None:
        for face in faces[1]:
            bbox = face[:4].astype(int)
            x, y, w, h = bbox
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image


def export_embeddings(folder_path, output_file):
    """
    Export the embeddings
    """

    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(filename)[0]
            img_path = os.path.join(folder_path, filename)
            img = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(img)

            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(name)

    # Save embeddings to a pickle file
    embeddings_data = {"encodings": known_face_encodings, "names": known_face_names}
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings_data, f)
        print(f"Embeddings and names saved to {output_file}")

def detect_characters_yunet(img, face_detector, known_face_encodings, known_face_names):

    h, w, _ = img.shape
    face_detector.setInputSize((w, h))

    # Perform face detection using YuNet
    faces = face_detector.detect(img)

    # Check if faces are detected
    if faces[1] is not None:
        for face in faces[1]:
            bbox = face[:4].astype(int)
            x, y, w, h = bbox

            # Crop the detected face
            face_image = img[y:y + h, x:x + w]

            # Convert face image to RGB for face_recognition
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            # Step 4: Encode the Detected Face
            encodings = get_embedding(rgb_face)
            name = "Unknown"
            if encodings:
                encoding = encodings[0]

                # Step 5: Compare with Known Faces
                distances = face_recognition.face_distance(known_face_encodings, encoding)
                best_match_index = np.argmin(distances)

                if distances[best_match_index] < 0.6:
                    name = known_face_names[best_match_index]

            # Step 6: Draw Bounding Box and Name
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return img

def get_embedding(img):
    """
    Extract the face embedding from an input image.
    """
    face_encodings = face_recognition.face_encodings(img)
    if face_encodings:
        return face_encodings

    return None

def load_embeddings(filename):
    """
    Load face embeddings and names from a pickle file.

    Parameters:
    - filename: Path to the pickle file containing the embeddings.
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    known_face_encodings = data.get("encodings", [])
    known_face_names = data.get("names", [])

    print(f"Successfully loaded {len(known_face_encodings)} embeddings from {filename}")
    return known_face_encodings, known_face_names


if __name__ == '__main__':
    folder_path = 'data'  # Replace with your folder path
    output_file = 'embeddings/face_embeddings.pkl'
    export_embeddings(folder_path, output_file)
    # known_face_encodings, known_face_names = load_embeddings(output_file)
    # print(known_face_encodings, known_face_names)