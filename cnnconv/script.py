import os
import numpy as np
import matplotlib.pyplot as plt
import pickle  # Añadir esta importación
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----------------------------------------------------------------
# CONFIGURACIÓN DEL SISTEMA
# ----------------------------------------------------------------
IMG_SIZE = 128 
CHANNELS = 3
BATCH_SIZE = 32  # Reducido para manejar mejor la memoria
EPOCHS = 45
LEARNING_RATE = 0.001

# Ruta al dataset
dataset_path = os.path.join(os.getcwd(), 'practica_2/animals-dataset/animals-dataset')

# ----------------------------------------------------------------
# PASO 1: Configuración de Generadores de Datos (Solución MemoryError)
# ----------------------------------------------------------------
print(f"Configurando generadores desde: {dataset_path}")

# Generador para entrenamiento con augmentación de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# Generador para test (solo normalización, sin augmentación)
test_datagen = ImageDataGenerator(rescale=1./255)

# Cargar datos de entrenamiento
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    interpolation='bilinear',      # Añadir método de interpolación
    keep_aspect_ratio=False        # Forzar redimensionamiento exacto, si es True mantiene proporciones y agrega borde negro
)

# Cargar datos de validación
validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    interpolation='bilinear',
    keep_aspect_ratio=False
)

# Obtener nombres de clases y número de clases
class_names = list(train_generator.class_indices.keys())
nClasses = len(class_names)

print(f"\nClases detectadas: {class_names}")
print(f"Número total de clases: {nClasses}")
print(f"Imágenes de entrenamiento: {train_generator.samples}")
print(f"Imágenes de validación: {validation_generator.samples}")

# ----------------------------------------------------------------
# PASO 2: Definición de la Arquitectura CNN
# ----------------------------------------------------------------
model = Sequential()

# --- BLOQUE 1: Características de bajo nivel (Bordes, Colores) --- 128
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)))
model.add(LeakyReLU(alpha=0.1)) # Evita neuronas muertas
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3)) # Regularización suave

# --- BLOQUE 2: Características medias (Texturas, Formas simples) --- 64
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))

# --- BLOQUE 3: Características complejas (Partes de animales) --- 32
# Aumentamos filtros a 128 para capturar más complejidad biológica
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))

# --- BLOQUE 4: Características avanzadas (Animales completos) --- 16
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3)) 

# --- CLASIFICADOR (Top Model) ---
model.add(Flatten())
model.add(Dense(512)) # Capas que tienen conexiones desde cada neurona previa hacia cada neurona actual
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))  # Regularización fuerte antes de la decisión final 
model.add(Dense(128))   
model.add(Dense(nClasses, activation='softmax'))

# Mostrar resumen
model.summary()

# ----------------------------------------------------------------
# PASO 3: Compilación y Entrenamiento con Generadores
# ----------------------------------------------------------------
optimizer = Adam(learning_rate=LEARNING_RATE) # Esto es lo que permite a la red saber como agustar los pesos y sesgos durante el entrenamiento de manera eficiente

model.compile(
    loss='categorical_crossentropy', 
    optimizer=optimizer,
    metrics=['accuracy']
)

# Callback para reducir learning rate cuando se estanque
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,              # Reduce LR a la mitad
    patience=3,              # Espera 3 épocas sin mejora
    min_lr=1e-7,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5,              
    min_delta=0.001,         # Mejora mínima requerida
    restore_best_weights=True,
    verbose=1
)

print("\nIniciando entrenamiento con generadores...")
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator), 
    epochs=EPOCHS,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=len(validation_generator), 
    callbacks=[reduce_lr, early_stop]  
)


print(f"\nMejor precisión en validación: {max(history.history['val_accuracy'])*100:.2f}%")

# ----------------------------------------------------------------
# PASO 5: Guardado del Modelo y del Historial
# ----------------------------------------------------------------
model_filename = "animal_classifier_optimized-2.h5"
model.save(model_filename)
print(f"\nModelo guardado exitosamente como: {model_filename}")

# Guardar el historial de entrenamiento
history_filename = "training_history.pkl"
with open(history_filename, 'wb') as f:
    pickle.dump(history.history, f)
print(f"Historial guardado como: {history_filename}")

# ----------------------------------------------------------------
# PASO 6: Visualización de Resultados
# ----------------------------------------------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# Guardar la figura antes de mostrarla
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
print(f"\nGráficas guardadas como: training_results.png")

plt.show()