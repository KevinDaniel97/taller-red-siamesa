import cv2
import os
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

padding = 50  # Ajustado para mejorar precisión

# Directorio de imágenes
image_dir = 'C:/Users/User/Downloads/taller-red-siamesa-vesion3f/taller-red-siamesa-vesion3f/dataset'

print("Carpetas en el directorio de imágenes:")
print(os.listdir(image_dir))  

# Mapeo de clases basado en carpetas
class_to_name = {0: "Desconocido"}
for idx, folder_name in enumerate(os.listdir(image_dir), start=1):
    partes = folder_name.split("_")
    user_name = partes[1] if len(partes) > 1 else "Desconocido"
    class_to_name[idx] = user_name

# Cargar el modelo de predicción
try:
    model_facial = load_model('C:/Users/User/Downloads/taller-red-siamesa-vesion3f/taller-red-siamesa-vesion3f/reconocimmientoFacial/redes_entrenadas/ReconocimientoFacialV04.h5')
    print("✅ Modelo cargado exitosamente.")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    exit()

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Inicializar la captura de video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

if not cap.isOpened():
    print("❌ Error: No se pudo abrir la cámara.")
    exit()

# 🔹 Variables para tracking
last_x, last_y, last_w, last_h = None, None, None, None
last_name, last_confidence = "Desconocido", 0
frame_count = 0

def reconocer_rostro(face):
    """Detecta y predice el rostro con el modelo cargado."""
    try:
        if face is None or face.size == 0:
            print("⚠️ Advertencia: La imagen de la cara está vacía o es None.")
            return last_name, last_confidence  

        face_resized = cv2.resize(face, (128, 128))
        face_resized = np.expand_dims(face_resized, axis=0).astype('float32') / 255.0  

        usuario = model_facial.predict(face_resized)
        probabilidad = np.max(usuario) * 100
        clase_predicha = np.argmax(usuario, axis=1)[0]

        user_name = class_to_name.get(clase_predicha, "Desconocido")

        if probabilidad < 50:  
            return last_name, last_confidence  

        print(f"✅ Usuario detectado: {user_name}, Confianza: {probabilidad:.2f}%")
        return user_name, probabilidad
    except Exception as e:
        print(f"❌ Error al predecir el modelo: {e}")
        return last_name, last_confidence  

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: No se pudo leer el fotograma.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # 🔹 Inicializar `nombre` y `confianza` al inicio de cada frame
    nombre, confianza = last_name, last_confidence  
    detected = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ih, iw, _ = frame.shape
            x_min, y_min, x_max, y_max = iw, ih, 0, 0

            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * iw), int(landmark.y * ih)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            # Definir el área del rostro con padding
            x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min
            x_centered = max(0, x - padding)
            y_centered = max(0, y - padding)
            w_centered = min(iw, x + w + padding) - x_centered
            h_centered = min(ih, y + h + padding) - y_centered

            face = frame[y_centered:y_centered + h_centered, x_centered:x_centered + w_centered]
            
            if face.shape[0] > 0 and face.shape[1] > 0:
                nombre, confianza = reconocer_rostro(face)
                confianza = round(confianza, 3)

                last_x, last_y, last_w, last_h = x, y, w, h
                last_name, last_confidence = nombre, confianza
                frame_count = 0
                detected = True

                # 🔥 Generar heatmap con bordes completamente rojos
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                # Detectar bordes con Canny
                edges = cv2.Canny(gray, 50, 150)

                # 🔹 Convertir los bordes en ROJO (BGR → (0, 0, 255))
                edges_colored = np.zeros_like(face)
                edges_colored[:, :, 2] = edges  # Solo afecta el canal rojo

                # 🔹 Menos oscurecimiento del fondo para más transparencia
                darkened_face = cv2.addWeighted(face, 0.8, np.zeros_like(face), 0.2, 0)

                # 🔹 Superponer los bordes ROJOS en la imagen oscura
                overlay = cv2.addWeighted(darkened_face, 0.8, edges_colored, 0.8, 0)
                frame[y_centered:y_centered + h_centered, x_centered:x_centered + w_centered] = overlay

    if not detected and frame_count < 10:  
        if last_x is not None:
            x, y, w, h = last_x, last_y, last_w, last_h
            nombre, confianza = last_name, last_confidence
        frame_count += 1

    if nombre is not None and confianza is not None:
        print(f"📢 Mostrando en pantalla: {nombre} ({confianza}%)")

    # Dibujar el rectángulo y el nombre en la imagen
    if last_x is not None:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Usuario: {nombre} ({confianza}%)", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('🔹 Reconocimiento Facial con Bordes Rojos', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
