import tkinter as tk
from tkinter import filedialog
import subprocess
import os
import RecortarProcesarImagen as gen

import os

# Función para abrir la carpeta
def open_folder():
    folder_name = label_entry.get()
    
    # Obtener todas las carpetas dentro de dataset/propias
    existing_folders = os.listdir("dataset")
    
    # Buscar la carpeta que coincida con el nombre proporcionado
    matching_folder = None
    for folder in existing_folders:
        if folder.endswith(f"_{folder_name}"):
            matching_folder = folder
            break
    
    if matching_folder:
        folder_path = os.path.join("dataset", matching_folder)
        os.startfile(folder_path)  # Esto abre la carpeta en el explorador de archivos
    else:
        print(f"Error: la carpeta con el nombre {folder_name} no existe.")

# Función para iniciar la captura
def start_capture():
    folder_name = label_entry.get()
    
    # Obtener todas las carpetas dentro de dataset/propias
    existing_folders = os.listdir("dataset")
    
    # Filtrar las carpetas que cumplen con el formato 001_Nombre
    numbered_folders = [f for f in existing_folders if f.split('_')[0].isdigit()]
    
    if numbered_folders:
        # Obtener el número más alto y sumarle uno
        highest_number = max([int(f.split('_')[0]) for f in numbered_folders])
        new_number = highest_number + 1
    else:
        # Si no hay carpetas, empezar con 001
        new_number = 1
    
    # Formatear el nuevo número con tres dígitos y agregar el nombre de la carpeta
    new_folder_name = f"{new_number:03d}_{folder_name}"
    folder_path = os.path.join("dataset", new_folder_name)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # Crear la carpeta si no existe
    
    gen.capture_faces(folder_path, 100)

# Configuración de la ventana
window = tk.Tk()
window.title("Interfaz de Captura de Imágenes")

# Cuadro de texto para el nombre de la carpeta
label_entry = tk.Entry(window, width=30)
label_entry.pack(pady=10)

# Botón para iniciar la captura
capture_button = tk.Button(window, text="Iniciar Captura", command=start_capture)
capture_button.pack(pady=10)

# Botón para abrir la carpeta
open_button = tk.Button(window, text="Abrir Carpeta", command=open_folder)
open_button.pack(pady=10)

# Iniciar la interfaz gráfica
window.mainloop()
