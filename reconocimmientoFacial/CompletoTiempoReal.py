import cv2
import os
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

padding = 100

# Directorio de las imágenes
image_dir = 'D:/taller/dataset'

print("Carpetas en el directorio de imágenes:")
print(os.listdir(image_dir))  # Esto mostrará todas las carpetas encontradas

# Crear un diccionario que mapea la clase 0 a "Desconocido"
class_to_name = {0: 'Desconocido'}

# Agregar las demás clases automáticamente
for idx, folder_name in enumerate(os.listdir(image_dir), start=1):
    print(f"Procesando carpeta: {folder_name}")  # Depuración para ver qué nombres se están procesando
    partes = folder_name.split('_')
    if len(partes) > 1:
        user_name = partes[1]
    else:
        print(f"⚠️ Advertencia: La carpeta '{folder_name}' no tiene un guion bajo ('_'). Se ignorará.")
        user_name = "Desconocido"
    class_to_name[idx] = user_name

# Cargar el modelo de predicción
model_facial = load_model('redes_entrenadas/ReconocimientoFacialV02.h5')

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Inicializar la captura de video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

def reconocer_rostro(face):
    try:
        face_resized = cv2.resize(face, (128, 128))
        face_resized = face_resized.astype('float32') / 255.0 
        face_resized = np.expand_dims(face_resized, axis=0)

        usuario = model_facial.predict(face_resized)
        probabilidad = np.max(usuario) * 100
        clase_predicha = np.argmax(usuario, axis=1)[0]

        user_name = class_to_name.get(clase_predicha, "Desconocido")
        if probabilidad < 70:
            user_name = "Desconocido"
            probabilidad = 0

        print(f"Usuario: {user_name}, Confianza: {probabilidad:.2f}%")
        return user_name, probabilidad
    except Exception as e:
        print(f"Error al predecir el modelo: {e}")
        return "Desconocido", 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el fotograma.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ih, iw, _ = frame.shape
            x_min = iw
            y_min = ih
            x_max = 0
            y_max = 0

            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * iw), int(landmark.y * ih)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min
            x_centered = max(0, x - padding)
            y_centered = max(0, y - 2 * padding)
            w_centered = min(iw, x + w + padding) - x_centered
            h_centered = min(ih, y + h + 2 * padding) - y_centered

            face = frame[y_centered:y_centered + h_centered, x_centered:x_centered + w_centered]
            if face.size > 0:
                nombre, confianza = reconocer_rostro(face)
                confianza = round(confianza, 3)

            cv2.putText(frame, f"Usuario: {nombre} ({confianza}%)", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Reconocimiento Facial', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
