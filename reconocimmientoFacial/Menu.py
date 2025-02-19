#pip install ttkbootstrap

import tkinter as tk
import ttkbootstrap as ttk
import subprocess
import os

# Obtener el directorio actual
script_dir = os.path.dirname(os.path.abspath(__file__))

# Función para ejecutar un script normalmente
def ejecutar_script(nombre_script):
    ruta_script = os.path.join(script_dir, nombre_script)
    if os.path.exists(ruta_script):
        subprocess.run(["python", ruta_script], check=True)
    else:
        print(f"Error: No se encontró el archivo {ruta_script}")

# Función para abrir un archivo en VS Code
def abrir_en_vscode(nombre_script):
    ruta_script = os.path.join(script_dir, nombre_script)
    print(f"Intentando abrir: {ruta_script}")

    if os.path.exists(ruta_script):
        subprocess.run(f'code "{ruta_script}"', shell=True, check=True)  # Manejo de espacios en la ruta
    else:
        print(f"❌ Error: No se encontró el archivo {ruta_script}")

# Crear ventana principal
root = ttk.Window(themename="morph")  
root.title("Reconocimiento Facial - Control")
root.geometry("500x350")  # Aumentamos el tamaño para más botones
root.resizable(False, False)

# Crear un Canvas para dibujar el fondo con degradado
canvas = tk.Canvas(root, width=500, height=350, highlightthickness=0)
canvas.pack(fill="both", expand=True)

# Dibujar el degradado
for i in range(350):
    color = "#{:02x}{:02x}{:02x}".format(
        min(255, max(0, int(5 + i * 0.2))),
        min(255, max(0, int(30 + i * 0.5))),
        min(255, max(0, int(120 + i * 0.6)))
    )
    canvas.create_line(0, i, 500, i, fill=color)

# Crear un frame encima del fondo
frame = ttk.Frame(root, padding=20)
canvas.create_window(250, 175, window=frame)  # Centrar frame

# Crear el texto en el Canvas **después** de crear el frame
titulo_id = canvas.create_text(250, 40, text="Menú Principal", font=("Helvetica", 18, "bold"), fill="white", anchor="center")

# Traer el título al frente
canvas.tag_raise(titulo_id)

# Botones con diseño moderno
botones = [
    ("📷 TiempoReal", lambda: ejecutar_script("TiempoReal.py")),
    ("📡 HealMaps", lambda: ejecutar_script("CompletoTiempoReal_HealMaps copy 2.py")),
    ("📂 Guardar Nuevo", lambda: ejecutar_script("GuardarNuevo.py")),
    ("🚀 Entrenamiento (VS Code)", lambda: ejecutar_script("EntrenamientoRF.py")),
    ("🔍 Predicción (VS Code)", lambda: abrir_en_vscode("Prediccion.ipynb")),
]

# Crear los botones en una cuadrícula 2x3
for i, (texto, comando) in enumerate(botones):
    row, col = divmod(i, 2)
    btn = ttk.Button(frame, text=texto, command=comando,
                     bootstyle="primary-outline", width=30, padding=10)
    btn.grid(row=row + 1, column=col, padx=10, pady=10)

# Ejecutar la interfaz gráfica
root.mainloop()
