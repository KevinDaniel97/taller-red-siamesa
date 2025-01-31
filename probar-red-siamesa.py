import cv2
import numpy as np
import tensorflow as tf
import bluetooth
from keras.models import load_model
from keras.saving import register_keras_serializable

# Registrar la función personalizada
@register_keras_serializable()
def euclidean_distance(vectors):
    vector1, vector2 = vectors
    sum_square = tf.reduce_sum(tf.square(vector1 - vector2), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))

def contrastive_loss(Y_true, D):
    margin = 1
    return tf.reduce_mean(Y_true * tf.square(D) + (1 - Y_true) * tf.maximum((margin - D), 0))

def accuracy(y_true, y_pred):
    return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.cast(y_pred < 0.5, y_true.dtype)), tf.float32))

# Cargar el modelo entrenado
model = load_model(
    'siamese_nn.keras',
    custom_objects={
        'euclidean_distance': euclidean_distance,
        'contrastive_loss': contrastive_loss,
        'accuracy': accuracy
    }
)

def write_on_frame(frame, text, text_x, text_y):
    (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)[0]
    box_coords = ((text_x, text_y), (text_x + text_width + 20, text_y - text_height - 20))
    cv2.rectangle(frame, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, text, (text_x, text_y - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return frame

# Conectar al dispositivo Bluetooth del celular
print("Buscando dispositivos Bluetooth...")
nearby_devices = bluetooth.discover_devices(duration=8, lookup_names=True)

if not nearby_devices:
    print("No se encontraron dispositivos Bluetooth. Asegúrate de que el celular está en modo visible.")
    exit()

print("Dispositivos encontrados:")
for idx, (addr, name) in enumerate(nearby_devices):
    print(f"{idx}: {name} - {addr}")

# Seleccionar el dispositivo
device_idx = int(input("Selecciona el índice del dispositivo: "))
device_addr = nearby_devices[device_idx][0]

# Crear conexión Bluetooth
port = 1  # Generalmente 1 es el puerto predeterminado
sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)

try:
    sock.connect((device_addr, port))
    print(f"Conectado a {nearby_devices[device_idx][1]}")
except Exception as e:
    print("Error al conectar Bluetooth:", e)
    exit()

# Onboarding: Capturar una imagen de referencia
print("Enviando solicitud para capturar imagen de referencia en el celular. Pulsa 's' en el celular.")
while True:
    data = sock.recv(1024)  # Recibe datos del celular
    if data and data.decode().strip() == "captura":
        img_bytes = sock.recv(50000)  # Recibir imagen
        np_arr = np.frombuffer(img_bytes, np.uint8)
        onboarding_img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        onboarding_img = cv2.resize(onboarding_img, (92, 112))
        onboarding_img = onboarding_img.astype('float32') / 255
        onboarding_img = onboarding_img.reshape(1, 112, 92, 1)
        break

print("Imagen de referencia capturada.")

# Reconocimiento facial en tiempo real desde el celular
print("Presiona 'q' en el celular para salir.")

while True:
    data = sock.recv(1024)
    if data and data.decode().strip() == "captura":
        img_bytes = sock.recv(50000)  # Recibir imagen
        np_arr = np.frombuffer(img_bytes, np.uint8)
        recognition_img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        recognition_img = cv2.resize(recognition_img, (92, 112))
        recognition_img = recognition_img.astype('float32') / 255
        recognition_img = recognition_img.reshape(1, 112, 92, 1)

        # Realizar predicción
        pred = model.predict([onboarding_img, recognition_img])[0][0]
        text = "si es men" if pred < 0.5 else "no es ese men"

        # Mostrar en consola (ya que no estamos usando ventana de OpenCV)
        print(text)

    if data and data.decode().strip() == "q":
        break

sock.close()
print("Conexión Bluetooth cerrada.")
