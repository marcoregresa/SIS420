#Laboratorio 2 - Redes Neuronales
#Nombre: Escobar Ruiz Marco Antonio
#Asignatura: Inteligencia Artificial SIS420
"""
Nombre del Dataset: Sign Language Gesture Images Dataset
URL: https://www.kaggle.com/datasets/ahmedkhanak1995/sign-language-gesture-images-dataset?select=Gesture+Image+Data
Descripcion: 
El dataset consta de 37 gestos diferentes de señas con las manos que incluyen gestos del alfabeto de la A a la Z,
gestos de números del 0 al 9 y también un gesto para el espacio que significa cómo las personas sordas o mudas 
representan el espacio entre dos letras o dos palabras mientras se comunican. El conjunto de datos tiene dos partes,
es decir, dos carpetas (1) - Datos de imágenes de gestos - que consisten en imágenes en color de las manos para diferentes gestos.
Cada imagen de gesto tiene un tamaño de 50X50 y se encuentra en su nombre de carpeta especificado que es Las carpetas de la A a la Z
consisten en imágenes de gestos de la A a la Z y las carpetas del 0 al 9 consisten en gestos del 0 al 9 respectivamente, la carpeta '_'
consiste en imágenes del gesto para el espacio. Cada gesto tiene 1500 imágenes, por lo que en total hay 37 gestos, lo que significa que hay 55,500 imágenes
para todos los gestos en la primera carpeta y en la segunda carpeta que es (2) - Datos preprocesados ​​de imágenes de gestos que tiene el mismo número de carpetas 
y el mismo número de imágenes que es 55,500. La diferencia radica en que estas imágenes son imágenes binarias de umbral convertidas para fines de entrenamiento y prueba. 
La red neuronal convolucional es ideal para este conjunto de datos, tanto para el entrenamiento de modelos como para la predicción de gestos.
"""
import os
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Ruta a la carpeta raíz que contiene las subcarpetas 'A' a 'Z' y '0' a '9' y '_'
root_dir = "Gesture Image Data"
# Tamaño de las imágenes
image_size = (50, 50)
# Extensiones válidas
valid_exts = ('.jpg', '.jpeg', '.png')
# Listas para datos y etiquetas
X = []
Y = []
# Crear un mapeo de etiquetas a números (0, 1, 2, ..., 36)
labels = sorted(os.listdir(root_dir))
label_map = {label: idx for idx, label in enumerate(labels)}

print("Cargando imágenes...")
for label in labels:
    folder = os.path.join(root_dir, label)
    if not os.path.isdir(folder): continue

    for fname in tqdm(os.listdir(folder), desc=f"Cargando '{label}'"):
        if fname.lower().endswith(valid_exts):
            try:
                img_path = os.path.join(folder, fname)
                img = Image.open(img_path).convert('L')  # Convertir a escala de grises para un entrenamiento mejor y mas reducido
                img = img.resize(image_size)
                img_array = np.array(img).flatten() / 255.0  # Normalizar
                X.append(img_array)# imagenes con  sus pixeles
                Y.append(label_map[label])# etiquetas - categorias de imagenes
            except Exception as e:
                print(f"Error al cargar {fname}: {e}")

# Convertir a arrays
X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.int64)

# Dividir datos en entrenamiento y prueba 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Convertir a tensores
X_train_t = torch.tensor(X_train)
X_test_t = torch.tensor(X_test)
Y_train_t = torch.tensor(Y_train)
Y_test_t = torch.tensor(Y_test)

# Crear el modelo
D_in = 50 * 50  # 2500
H_1, H_2, H_3 = 512, 256, 128 #capas ocultas o intermedias
D_out = len(label_map)  # 37 etiquetas (A-Z + 0-9 + _)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H_1),
    torch.nn.ReLU(),
    torch.nn.Linear(H_1, H_2),
    torch.nn.ReLU(),
    torch.nn.Linear(H_2, H_3),
    torch.nn.ReLU(),
    torch.nn.Linear(H_3, D_out)
)

# Configurar pérdida y optimizador
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # hiperparametro

# Entrenamiento
epochs = 2000
log_each = 100
losses = []

print("Entrenando modelo...")
for e in range(1, epochs + 1):
    model.train()
    y_pred = model(X_train_t) # entrenando
    loss = criterion(y_pred, Y_train_t) # comparando el entrenamiento
    losses.append(loss.item()) # guardando perdidas, cual alejado esta de etiquetas reales

    optimizer.zero_grad() # optimizar gradientes
    loss.backward() # recalcular delante y atras
    optimizer.step() #Actualiza los pesos del modelo con los gradientes calculados y según el optimizador

    if e % log_each == 0:
        print(f"Epoch {e}/{epochs} | Loss promedio: {np.mean(losses):.5f}")
        losses = []

# Evaluación
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_t).argmax(dim=1).numpy() # evaluando modelo

accuracy = accuracy_score(Y_test, y_pred_test) # calculando la precision del modelo con los datos de prueba
print(f"Precisión en conjunto de prueba: {accuracy:.5f}") 

# Guardardon el modelo
torch.save(model.state_dict(), "modelo_gesture.pth")
print("Modelo guardado como modelo_gesture.pth")