import os
import cv2
import time
import threading
import mediapipe as mp

def capture_faces_thread(frame_queue, output_dir, max_images, padding, stop_event):
    mp_face_detection = mp.solutions.face_detection
    captured_images = 0

    os.makedirs(output_dir, exist_ok=True)

    with mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_detection:
        while captured_images < max_images and not stop_event.is_set():
            if not frame_queue:
                continue

            frame = frame_queue.pop(0)  # Obtener el último frame disponible
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    # Aplicar padding
                    x = max(0, x - 3 * padding)
                    y = max(0, y - 4 * padding)
                    w = min(iw - x, w + 5 * padding)
                    h = min(ih - y, h + 8 * padding)

                    # Calcular la posición de los ojos
                    eye_x = x + int(w * 0.5)  # Aproximadamente el centro de la cara en el eje X
                    eye_y = y + int(h * 0.4)  # Aproximadamente el 40% desde la parte superior de la cara

                    # Ajustar el recorte para centrar en los ojos
                    new_x = max(0, eye_x - int(w / 2))
                    new_y = max(0, eye_y - int(h / 2))
                    new_w = min(iw - new_x, w)
                    new_h = min(ih - new_y, h)

                    face = frame[new_y:new_y + new_h, new_x:new_x + new_w]
                    if face.size > 0:
                        img_path = os.path.join(output_dir, f"face_{str(captured_images).zfill(4)}.jpg")
                        cv2.imwrite(img_path, face)
                        captured_images += 1
                        print(f"Imagen capturada: {img_path}")
                        time.sleep(0.5)
                        if captured_images >= max_images:
                            stop_event.set()
                            break

    print(f"Captura completada: {captured_images} imágenes guardadas en {output_dir}")

def show_camera(frame_queue, width, height, fps, stop_event):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

        # Ajustar parámetros adicionales para mejorar la imagen
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)  # Ajusta el brillo (valor entre 0 y 1)
    cap.set(cv2.CAP_PROP_CONTRAST, 0.5)    # Ajusta el contraste (valor entre 0 y 1)
    cap.set(cv2.CAP_PROP_SATURATION, 0.5)  # Ajusta la saturación (valor entre 0 y 1)
    cap.set(cv2.CAP_PROP_GAIN, 0.5)        # Ajusta la ganancia (valor entre 0 y 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, -4) 

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Error al acceder a la cámara.")
            break

        # Invertir la imagen horizontalmente
        frame = cv2.flip(frame, 1)

        # Agregar el frame a la cola
        if len(frame_queue) < 5:  # Limitar el tamaño de la cola
            frame_queue.append(frame)

        # Mostrar la cámara
        cv2.imshow('Capturando Rostros', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()

def capture_faces(output_dir='dataset/person_0', max_images=100, padding=50, width=720, height=1280, fps=60):
    frame_queue = []
    stop_event = threading.Event()
    camera_thread = threading.Thread(target=show_camera, args=(frame_queue, width, height, fps, stop_event))
    capture_thread = threading.Thread(target=capture_faces_thread, args=(frame_queue, output_dir, max_images, padding, stop_event))

    camera_thread.start()
    capture_thread.start()

    camera_thread.join()
    capture_thread.join()

# Llamada a la función
# capture_faces()  # Descomenta esta línea para ejecutar la función