import cv2

def get_camera_properties():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    # Listar algunas propiedades comunes
    properties = {
        "FRAME_WIDTH": cv2.CAP_PROP_FRAME_WIDTH,
        "FRAME_HEIGHT": cv2.CAP_PROP_FRAME_HEIGHT,
        "FPS": cv2.CAP_PROP_FPS,
        "BRIGHTNESS": cv2.CAP_PROP_BRIGHTNESS,
        "CONTRAST": cv2.CAP_PROP_CONTRAST,
        "SATURATION": cv2.CAP_PROP_SATURATION,
        "GAIN": cv2.CAP_PROP_GAIN,
        "EXPOSURE": cv2.CAP_PROP_EXPOSURE
    }

    for prop_name, prop_id in properties.items():
        value = cap.get(prop_id)
        print(f"{prop_name}: {value}")

    cap.release()

get_camera_properties()