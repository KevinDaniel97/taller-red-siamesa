import cv2
import os
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

padding = 100

# 🔹 Directorio de las imágenes
image_dir = 'D:/taller/dataset'

# 🔹 Leer y ordenar las carpetas numéricamente para garantizar que coincidan con el entrenamiento
fotos_mias_folders = sorted(
    [folder for folder in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, folder))],
    key=lambda x: int(x.split("_")[0])  # Ordenar numéricamente por el prefijo del nombre
)

print("📂 Carpetas ordenadas en el directorio de imágenes:")
print(fotos_mias_folders)

# 🔹 Crear un diccionario que mapea los índices del modelo con los nombres de usuarios
class_to_name = {i: folder.split('_', 1)[1] if '_' in folder else folder for i, folder in enumerate(fotos_mias_folders)}
print("\n🔍 Mapeo de Clases Correcto:", class_to_name)

# 🔹 Cargar el modelo de predicción
model_facial = load_model('D:/taller/reconocimmientoFacial/redes_entrenadas/ReconocimientoFacialV02.h5')

# 🔹 Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 🔹 Inicializar la captura de video
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

        # 🔹 Verificar que la clase predicha existe en `class_to_name`
        user_name = class_to_name.get(clase_predicha, "Desconocido")

        # 🔹 Depuración: Verificar salida del modelo
        print(f"🔹 Predicción: {usuario}")
        print(f"🔹 Clase predicha: {clase_predicha}, Nombre asignado: {user_name}, Confianza: {probabilidad:.2f}%")

        if probabilidad < 50:
            user_name = "Desconocido"
            probabilidad = 0

        return user_name, probabilidad
    except Exception as e:
        print(f"⚠️ Error al predecir el modelo: {e}")
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
