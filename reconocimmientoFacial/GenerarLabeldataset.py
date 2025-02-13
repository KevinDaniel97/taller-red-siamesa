import tkinter as tk
from tkinter import ttk, messagebox
import os
import cv2
import time
import mediapipe as mp
import numpy as np

DATASET_PATH = "D:/taller/dataset"

# 🔹 Inicializar MediaPipe para detección facial
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# 🔹 Función para detectar y recortar el rostro en la imagen
def detectar_y_recortar_rostro(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)
    
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = (
                int(bboxC.xmin * iw),
                int(bboxC.ymin * ih),
                int(bboxC.width * iw),
                int(bboxC.height * ih),
            )
            
            # 🔹 Aplicar padding para mejorar el recorte
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(iw - x, w + 2 * padding)
            h = min(ih - y, h + 2 * padding)

            # 🔹 Extraer la región de la cara
            face = frame[y:y + h, x:x + w]
            
            # Si el rostro es válido, procesarlo
            if face.size > 0:
                # Redimensionar a 128x128
                face = cv2.resize(face, (128, 128))
                return face
    return None

# 🔹 Función para capturar imágenes con detección de rostros
def capture_images():
    folder_name = entry_folder.get().strip()
    if not folder_name:
        messagebox.showwarning("Advertencia", "Por favor, ingrese un nombre para la carpeta.")
        return

    existing_folders = os.listdir(DATASET_PATH)
    numbered_folders = [f for f in existing_folders if f.split('_')[0].isdigit()]
    new_number = max([int(f.split('_')[0]) for f in numbered_folders], default=0) + 1
    formatted_folder_name = f"{new_number:03d}_{folder_name}"
    folder_path = os.path.join(DATASET_PATH, formatted_folder_name)
    os.makedirs(folder_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "No se pudo abrir la cámara.")
        return

    messagebox.showinfo("Instrucciones", "Presione 'C' para comenzar la captura de imágenes.")

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "No se pudo capturar imagen.")
            break

        cv2.imshow("Presiona 'C' para iniciar la captura", frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

    count = 0
    while count < 40:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "No se pudo capturar imagen.")
            break

        # 🔹 Detectar y recortar rostro
        face = detectar_y_recortar_rostro(frame)
        if face is not None:
            img_path = os.path.join(folder_path, f"{count:03d}_{folder_name}.jpg")
            cv2.imwrite(img_path, face)
            count += 1
            print(f"Imagen {count}/40 capturada")

        cv2.imshow("Capturando imágenes", frame)
        time.sleep(0.1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 🔹 Función para abrir la carpeta donde se guardaron las imágenes
def open_folder():
    folder_name = entry_folder.get().strip()
    if not folder_name:
        messagebox.showwarning("Advertencia", "Por favor, ingrese un nombre de carpeta.")
        return

    existing_folders = os.listdir(DATASET_PATH)
    matching_folder = next((folder for folder in existing_folders if folder.endswith(f"_{folder_name}")), None)

    if matching_folder:
        folder_path = os.path.join(DATASET_PATH, matching_folder)
        os.startfile(folder_path)
    else:
        messagebox.showerror("Error", f"La carpeta '{folder_name}' no existe.")

# 🔹 Configuración de la ventana
window = tk.Tk()
window.title("Captura de Imágenes")
window.geometry("400x350")
window.resizable(False, False)

# 🔹 Marco principal
frame = ttk.Frame(window, padding=20)
frame.pack(expand=True)

# 🔹 Etiqueta y entrada de texto
ttk.Label(frame, text="Nombre de la Carpeta:").pack(anchor="w")
entry_folder = ttk.Entry(frame, width=30)
entry_folder.pack(pady=5)

# 🔹 Botón para capturar imágenes
btn_capture_images = ttk.Button(frame, text="Capturar Rostros", command=capture_images)
btn_capture_images.pack(pady=5, fill="x")

# 🔹 Botón para abrir la carpeta
btn_open = ttk.Button(frame, text="Abrir Carpeta", command=open_folder)
btn_open.pack(pady=5, fill="x")

# 🔹 Iniciar la interfaz gráfica
window.mainloop()
