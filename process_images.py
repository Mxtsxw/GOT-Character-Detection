import json
import os
import cv2
import numpy as np
import dlib

# Charger le détecteur de visages et le prédicteur de points de repère (landmarks)
shape_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')  # Modèle pour les points de repère
#dlib_face_recognition_model = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')  # Modèle de reconnaissance faciale
model = cv2.dnn.readNetFromONNX('models/arcfaceresnet100-8.onnx')


# Fonction pour extraire les descripteurs faciaux à partir d'une image
# def get_face_embeddings(image_path):
#     # Lire l'image
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Erreur: Impossible de charger l'image {image_path}.")
#         return None
#
#     # Convertir l'image en niveaux de gris
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Détecter les visages dans l'image
#     detector = dlib.get_frontal_face_detector()
#     faces = detector(gray)
#
#     embeddings = []
#     for face in faces:
#         # Extraire les points de repère du visage
#         shape = shape_predictor(gray, face)
#
#         # Calculer l'embedding du visage
#         face_embedding = dlib_face_recognition_model.compute_face_descriptor(image, shape)
#         embeddings.append(np.array(face_embedding))
#
#     return embeddings

def get_face_embeddings(image_path):
    # Lire l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur: Impossible de charger l'image {image_path}.")
        return None

    # Détecter les visages dans l'image
    detector = dlib.get_frontal_face_detector()
    faces = detector(image)

    embeddings = []
    for face in faces:
        # Extraire la région du visage
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_crop = image[y:y + h, x:x + w]

        # Prétraitement pour ArcFace
        try:
            face_crop = cv2.resize(face_crop, (112, 112))  # Dimension d'entrée attendue par ArcFace
            face_crop = face_crop.astype(np.float32)
            face_crop = face_crop / 255.0  # Normalisation
            blob = cv2.dnn.blobFromImage(face_crop, 1.0, (112, 112), (0, 0, 0), swapRB=True)
        except Exception as e:
            print(f"Erreur lors du prétraitement de l'image: {e}")
            continue

        # Passer l'image au modèle
        model.setInput(blob)
        embedding = model.forward()

        # Ajouter l'embedding à la liste (flatten pour obtenir un tableau 1D si nécessaire)
        embeddings.append(embedding.flatten())

    return embeddings

# Fonction pour enregistrer les descripteurs dans un fichier JSON
def save_face_embeddings(face_embeddings, name, filename='face_embeddings.json'):
    # Charger les descripteurs existants (s'il y en a)
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    # Ajouter ou mettre à jour les descripteurs pour la personne donnée
    data[name] = face_embeddings.tolist()

    # Enregistrer dans le fichier JSON
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


# Fonction principale pour traiter les images d'une personne et enregistrer les descripteurs
def process_images(images_folder):
    for image_name in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_name)

        # Extraire les descripteurs du visage
        embeddings = get_face_embeddings(image_path)

        # Si des descripteurs sont trouvés, les enregistrer
        if embeddings:
            person_name = os.path.splitext(image_name)[0]  # Utiliser le nom de l'image comme nom de la personne
            save_face_embeddings(embeddings[0], person_name)
            print(f"Descripteur pour {person_name} enregistré.")
        else:
            print(f"Aucun visage détecté dans {image_name}. Ignoré.")


# Dossier contenant les images des personnes
images_folder = 'images_personnes'  # Répertoire où les images des personnes sont stockées
process_images(images_folder)
