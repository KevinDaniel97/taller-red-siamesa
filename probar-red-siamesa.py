# probar-red-siamesa.py
import cv2
import numpy as np
import tensorflow as tf
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

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Onboarding: Capturar una imagen de referencia
print("Presiona 's' para capturar la imagen de referencia.")
while True:
    ret, frame = cap.read()
    cv2.imshow('Onboarding', frame)
    if cv2.waitKey(20) & 0xFF == ord('s'):
        onboarding_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        onboarding_img = cv2.resize(onboarding_img, (92, 112))
        onboarding_img = onboarding_img.astype('float32') / 255
        onboarding_img = onboarding_img.reshape(1, 112, 92, 1)
        break

cv2.destroyAllWindows()

# Reconocimiento facial en tiempo real
print("Presiona 'q' para salir.")
while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    recognition_img = cv2.resize(gray_frame, (92, 112))
    recognition_img = recognition_img.astype('float32') / 255
    recognition_img = recognition_img.reshape(1, 112, 92, 1)

    # Realizar predicción
    pred = model.predict([onboarding_img, recognition_img])[0][0]
    if pred < 0.5:
        text = "si es men"
    else:
        text = "no es ese men"

    # Mostrar el resultado en el frame
    frame = write_on_frame(frame, text, 50, 50)
    cv2.imshow('Reconocimiento Facial', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()