import os
import cv2
import numpy as np
import random
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.saving import register_keras_serializable
from multiprocessing import Pool, cpu_count

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

def create_pairs(X, Y, num_classes):
    pairs, labels = [], []
    class_idx = [np.where(Y == i)[0] for i in range(num_classes)]
    min_images = min(len(class_idx[i]) for i in range(num_classes)) - 1

    for c in range(num_classes):
        for n in range(min_images):
            img1 = X[class_idx[c][n]]
            img2 = X[class_idx[c][n + 1]]
            pairs.append((img1, img2))
            labels.append(1)

            neg_list = list(range(num_classes))
            neg_list.remove(c)
            neg_c = random.sample(neg_list, 1)[0]
            img1 = X[class_idx[c][n]]
            img2 = X[class_idx[neg_c][n]]
            pairs.append((img1, img2))
            labels.append(0)

    return np.array(pairs), np.array(labels)

def create_shared_network(input_shape):
    model = Sequential(name='Shared_Conv_Network')
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=128, activation='sigmoid'))
    return model

def process_image(file, folder, target_size):
    img = load_img(folder + "/" + file, color_mode='grayscale')
    img = img_to_array(img)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255
    img = img.reshape(target_size[0], target_size[1], 1)
    return img

def get_data(dir, target_size=(112, 92)):
    X_train, Y_train = [], []
    X_test, Y_test = [], []
    subfolders = sorted([file.path for file in os.scandir(dir) if file.is_dir()])
    
    with Pool(cpu_count()) as pool:
        for idx, folder in enumerate(subfolders):
            files = sorted(os.listdir(folder))
            images = pool.starmap(process_image, [(file, folder, target_size) for file in files])
            
            if idx < 35:
                X_train.extend(images)
                Y_train.extend([idx] * len(images))
            else:
                X_test.extend(images)
                Y_test.extend([idx - 35] * len(images))

    return (np.array(X_train), np.array(Y_train)), (np.array(X_test), np.array(Y_test))

# Entrenamiento del modelo
if __name__ == '__main__':
    # Desactivar oneDNN custom operations si es necesario
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    faces_dir = 'C:/Users/pc/Documents/taller_red_deep/faces_dir/archive'  # Ajusta la ruta a tu directorio de imágenes
    (X_train, Y_train), (X_test, Y_test) = get_data(faces_dir)
    num_classes = len(np.unique(Y_train))
    input_shape = X_train.shape[1:]
    shared_network = create_shared_network(input_shape)
    input_top = Input(shape=input_shape)
    input_bottom = Input(shape=input_shape)
    output_top = shared_network(input_top)
    output_bottom = shared_network(input_bottom)
    distance = Lambda(euclidean_distance, output_shape=(1,))([output_top, output_bottom])
    model = Model(inputs=[input_top, input_bottom], outputs=distance)
    training_pairs, training_labels = create_pairs(X_train, Y_train, num_classes=num_classes)

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True
    )

    # Función para aplicar aumento de datos
    def augment_images(images, labels, datagen, batch_size=32):
        augmented_images, augmented_labels = [], []
        total_images = len(images)
        steps = int(np.ceil(total_images / batch_size))  # Calcular el número de pasos necesarios

        # Crear un generador de datos aumentados
        data_gen = datagen.flow(images, labels, batch_size=batch_size, shuffle=False)

        for _ in range(steps):  # Iterar solo el número de pasos necesarios
            x, y = next(data_gen)  # Obtener el siguiente lote de datos aumentados
            augmented_images.append(x)
            augmented_labels.append(y)

        return np.concatenate(augmented_images), np.concatenate(augmented_labels)

    # Aplicar aumento de datos
    augmented_X_train, augmented_Y_train = augment_images(X_train, Y_train, datagen)

    # Compilar el modelo
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss=contrastive_loss, optimizer=optimizer, metrics=[accuracy])

    # Entrenar el modelo
    model.fit([training_pairs[:, 0], training_pairs[:, 1]], training_labels, batch_size=64, epochs=20)

    # Guardar el modelo entrenado
    model.save('siamese_nn.keras')
    print("Modelo entrenado y guardado como 'siamese_nn.keras'.")