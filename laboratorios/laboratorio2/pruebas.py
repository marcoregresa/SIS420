import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from PIL import Image
from sklearn.model_selection import train_test_split

# Parámetros
carpeta_base = "Gesture Image Data"
imagen_dim = (50, 50)
clases = sorted(os.listdir(carpeta_base))  # Etiquetas desde subcarpetas (A-Z, 0-9,-)
label_map = {label: i for i, label in enumerate(clases)}

# Cargar imágenes y etiquetas 
imagenes = [] #X
etiquetas = [] #Y

for clase in clases:
    carpeta_clase = os.path.join(carpeta_base, clase)
    for archivo in os.listdir(carpeta_clase):
        try:
            ruta = os.path.join(carpeta_clase, archivo)
            img = Image.open(ruta).convert("L")  # Convertir a escala de grises
            img = img.resize(imagen_dim)
            img_array = np.array(img).flatten()
            imagenes.append(img_array)
            etiquetas.append(label_map[clase])
        except Exception as e:
            print(f"Error al cargar {archivo}: {e}")

imagenes = np.array(imagenes) / 255.0 #X
etiquetas = np.array(etiquetas)       #Y 
  
# Dividir el conjunto en datos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(imagenes, etiquetas, test_size=0.2, random_state=42, stratify=etiquetas)

# Convertir a tensores
X_test_t = torch.tensor(X_test, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test, dtype=torch.long)

# Red neuronal
D_in = 2500  # 50x50 
H_1, H_2, H_3 = 512, 256, 128
D_out = len(label_map) # 37 etiquetas

modelo_cargado = torch.nn.Sequential(
    torch.nn.Linear(D_in, H_1),
    torch.nn.ReLU(),
    torch.nn.Linear(H_1, H_2),
    torch.nn.ReLU(),
    torch.nn.Linear(H_2, H_3),
    torch.nn.ReLU(),
    torch.nn.Linear(H_3, D_out)
)

# Cargar pesos
modelo_cargado.load_state_dict(torch.load("modelo_gesture.pth")) #cargando el modelo entrenado
modelo_cargado.eval()

# Visualizar una predicción
indice = random.randint(0, len(X_test) - 1) # escoge una imagen aleatoria
ejemplo = X_test[indice].reshape(50, 50) # reescalando
etiqueta_real = Y_test[indice] #obteniendo la etiqueta real


ejemplo_tensor = torch.tensor(X_test[indice], dtype=torch.float32).unsqueeze(0) #convierte en tensor
prediccion = modelo_cargado(ejemplo_tensor).argmax(dim=1).item() # obtiene la prediccion del ejemplo

# Obtener etiquetas en texto
inv_label_map = {v: k for k, v in label_map.items()} #etiquetas de numeros a texto original
etiqueta_real_txt = inv_label_map[etiqueta_real] # etiqueta real a texto
prediccion_txt = inv_label_map[prediccion] # prediccion real a texto

# Mostrar imagen
plt.imshow(ejemplo, cmap="gray")
plt.title(f"Etiqueta real: {etiqueta_real_txt} | Predicción: {prediccion_txt}", fontsize=14)
plt.axis("off")
plt.show()
