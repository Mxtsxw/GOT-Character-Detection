import os
import face_recognition

# Step 1: Load and Encode Known Faces
def load_known_faces(known_faces_dir='known_faces'):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(filename)[0]
            img_path = os.path.join(known_faces_dir, filename)
            img = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(img)

            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(name)

    return known_face_encodings, known_face_names


if __name__ == '__main__':
    known_face_encodings, known_face_names = load_known_faces('data/')
    print(known_face_encodings, known_face_names)