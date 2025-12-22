import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class NeuralNetworkTrainer:
    """
    Sistema completo de entrenamiento para clasificador de animales basado en CNN.
    
    Esta clase encapsula todo el proceso de entrenamiento incluyendo:
    - Carga y preprocesamiento de datos
    - Construcción de arquitectura CNN
    - Entrenamiento con callbacks adaptativos
    - Persistencia de modelos y métricas
    - Visualización de resultados
    """
    
    # ===== CONFIGURACIÓN DE HIPERPARÁMETROS =====
    INPUT_DIMENSIONS = 128          # Resolución de entrada para la red
    COLOR_CHANNELS = 3              # RGB (3 canales de color)
    SAMPLES_PER_BATCH = 32          # Tamaño de lote para entrenamiento
    TRAINING_ITERATIONS = 45        # Número máximo de épocas
    INITIAL_LEARNING_RATE = 0.001   # Tasa de aprendizaje inicial
    VALIDATION_SPLIT_RATIO = 0.2    # 20% de datos para validación
    
    # Configuración de regularización
    DROPOUT_LIGHT = 0.3             # Dropout suave en capas convolucionales
    DROPOUT_HEAVY = 0.5             # Dropout agresivo antes de clasificación
    
    # Configuración de callbacks
    LR_REDUCTION_FACTOR = 0.5       # Factor de reducción del learning rate
    EARLY_STOP_PATIENCE = 5         # Épocas de paciencia para early stopping
    LR_PLATEAU_PATIENCE = 3         # Épocas de paciencia para reducir LR
    
    def __init__(self, dataset_directory):
        """
        Inicializa el sistema de entrenamiento.
        
        Args:
            dataset_directory (str): Ruta al directorio que contiene las carpetas
                                     de clases con las imágenes de entrenamiento
        """
        self.dataset_root = dataset_directory
        self.neural_model = None
        self.training_history = None
        self.category_labels = None
        self.total_categories = 0
        self.train_data_flow = None
        self.validation_data_flow = None
        
        print(f"\n{'='*70}")
        print(f"  INICIALIZANDO SISTEMA DE ENTRENAMIENTO CNN")
        print(f"{'='*70}")
        print(f"Dataset: {self.dataset_root}")
    
    def setup_data_generators(self):
        """
        Configura los generadores de datos con augmentación y normalización.
        
        La augmentación de datos incluye:
        - Rotaciones aleatorias (±20°)
        - Desplazamientos horizontales/verticales (±20%)
        - Volteo horizontal (flip)
        - Zoom aleatorio (±20%)
        
        Esto previene overfitting al aumentar artificialmente la variedad del dataset.
        """
        print(f"\n[PASO 1] Configurando pipeline de datos...")
        
        # Pipeline de augmentación para datos de entrenamiento
        # Cada imagen se transforma aleatoriamente en cada época
        augmentation_pipeline = ImageDataGenerator(
            rescale=1./255,                          # Normaliza píxeles a [0, 1]
            validation_split=self.VALIDATION_SPLIT_RATIO,
            rotation_range=20,                       # Rotación aleatoria ±20°
            width_shift_range=0.2,                   # Desplazamiento horizontal ±20%
            height_shift_range=0.2,                  # Desplazamiento vertical ±20%
            horizontal_flip=True,                    # Espejo horizontal aleatorio
            zoom_range=0.2                           # Zoom aleatorio ±20%
        )
        
        # Pipeline simple para validación (solo normalización)
        validation_pipeline = ImageDataGenerator(rescale=1./255)
        
        # Crear flujo de datos de entrenamiento desde directorios
        self.train_data_flow = augmentation_pipeline.flow_from_directory(
            self.dataset_root,
            target_size=(self.INPUT_DIMENSIONS, self.INPUT_DIMENSIONS),
            batch_size=self.SAMPLES_PER_BATCH,
            class_mode='categorical',               # One-hot encoding para múltiples clases
            subset='training',                      # Usar el 80% para entrenamiento
            shuffle=True,                           # Mezclar muestras en cada época
            interpolation='bilinear',               # Método de redimensionamiento suave
            keep_aspect_ratio=False                 # Forzar tamaño exacto (puede deformar)
        )
        
        # Crear flujo de datos de validación
        self.validation_data_flow = augmentation_pipeline.flow_from_directory(
            self.dataset_root,
            target_size=(self.INPUT_DIMENSIONS, self.INPUT_DIMENSIONS),
            batch_size=self.SAMPLES_PER_BATCH,
            class_mode='categorical',
            subset='validation',                    # Usar el 20% para validación
            shuffle=False,                          # No mezclar validación (reproducibilidad)
            interpolation='bilinear',
            keep_aspect_ratio=False
        )
        
        # Extraer metadatos del dataset
        self.category_labels = list(self.train_data_flow.class_indices.keys())
        self.total_categories = len(self.category_labels)
        
        # Reportar estadísticas del dataset
        print(f"\n{'─'*70}")
        print(f"  ESTADÍSTICAS DEL DATASET")
        print(f"{'─'*70}")
        print(f"📁 Categorías detectadas: {self.category_labels}")
        print(f"🔢 Total de categorías: {self.total_categories}")
        print(f"🎯 Muestras de entrenamiento: {self.train_data_flow.samples}")
        print(f"✅ Muestras de validación: {self.validation_data_flow.samples}")
        print(f"{'─'*70}\n")
    
    def construct_network_architecture(self):
        """
        Construye la arquitectura de la Red Neuronal Convolucional.
        
        Arquitectura de 4 bloques convolucionales con complejidad creciente:
        
        Bloque 1 (128x128 → 64x64): Detecta características básicas
            - 32 filtros: Detecta bordes, colores, gradientes simples
            
        Bloque 2 (64x64 → 32x32): Detecta patrones de nivel medio
            - 64 filtros: Detecta texturas, formas geométricas básicas
            
        Bloque 3 (32x32 → 16x16): Detecta estructuras complejas
            - 128 filtros: Detecta partes de animales (orejas, patas, colas)
            
        Bloque 4 (16x16 → 8x8): Detecta objetos completos
            - 128 filtros: Detecta animales completos y contextos
        
        Clasificador final: Capas densas para la decisión
            - 512 neuronas: Integración de características
            - 128 neuronas: Abstracción final
            - N neuronas: Probabilidades por categoría (softmax)
        """
        print(f"[PASO 2] Construyendo arquitectura de red neuronal...")
        
        self.neural_model = Sequential(name='AnimalClassifierCNN')
        
        # ═════════════════════════════════════════════════════════
        # BLOQUE CONVOLUCIONAL 1: Extracción de características primitivas
        # ═════════════════════════════════════════════════════════
        # Resolución: 128x128 → 64x64
        self.neural_model.add(Conv2D(
            filters=32,                              # 32 detectores de patrones
            kernel_size=(3, 3),                      # Ventana de análisis 3x3
            padding='same',                          # Mantiene dimensiones
            input_shape=(self.INPUT_DIMENSIONS, self.INPUT_DIMENSIONS, self.COLOR_CHANNELS),
            name='conv_layer_1_basic_features'
        ))
        self.neural_model.add(LeakyReLU(alpha=0.1, name='activation_1'))  # Evita "neuronas muertas"
        self.neural_model.add(MaxPooling2D(pool_size=(2, 2), name='pooling_1'))  # Reduce a la mitad
        self.neural_model.add(Dropout(rate=self.DROPOUT_LIGHT, name='dropout_1'))  # Previene overfitting
        
        # ═════════════════════════════════════════════════════════
        # BLOQUE CONVOLUCIONAL 2: Patrones de nivel medio
        # ═════════════════════════════════════════════════════════
        # Resolución: 64x64 → 32x32
        self.neural_model.add(Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            name='conv_layer_2_textures'
        ))
        self.neural_model.add(LeakyReLU(alpha=0.1, name='activation_2'))
        self.neural_model.add(MaxPooling2D(pool_size=(2, 2), name='pooling_2'))
        self.neural_model.add(Dropout(rate=self.DROPOUT_LIGHT, name='dropout_2'))
        
        # ═════════════════════════════════════════════════════════
        # BLOQUE CONVOLUCIONAL 3: Estructuras complejas
        # ═════════════════════════════════════════════════════════
        # Resolución: 32x32 → 16x16
        # Incremento a 128 filtros para capturar mayor complejidad biológica
        self.neural_model.add(Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding='same',
            name='conv_layer_3_parts'
        ))
        self.neural_model.add(LeakyReLU(alpha=0.1, name='activation_3'))
        self.neural_model.add(MaxPooling2D(pool_size=(2, 2), name='pooling_3'))
        self.neural_model.add(Dropout(rate=self.DROPOUT_LIGHT, name='dropout_3'))
        
        # ═════════════════════════════════════════════════════════
        # BLOQUE CONVOLUCIONAL 4: Objetos completos
        # ═════════════════════════════════════════════════════════
        # Resolución: 16x16 → 8x8
        self.neural_model.add(Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding='same',
            name='conv_layer_4_objects'
        ))
        self.neural_model.add(LeakyReLU(alpha=0.1, name='activation_4'))
        self.neural_model.add(MaxPooling2D(pool_size=(2, 2), name='pooling_4'))
        self.neural_model.add(Dropout(rate=self.DROPOUT_LIGHT, name='dropout_4'))
        
        # ═════════════════════════════════════════════════════════
        # CLASIFICADOR DENSO: Toma de decisión final
        # ═════════════════════════════════════════════════════════
        self.neural_model.add(Flatten(name='flatten'))  # Convierte mapas 2D a vector 1D
        
        # Capa de integración: Combina todas las características extraídas
        self.neural_model.add(Dense(units=512, name='dense_integration'))
        self.neural_model.add(LeakyReLU(alpha=0.1, name='activation_integration'))
        self.neural_model.add(Dropout(rate=self.DROPOUT_HEAVY, name='dropout_heavy'))  # Regularización agresiva
        
        # Capa de abstracción: Representación de alto nivel
        self.neural_model.add(Dense(units=128, name='dense_abstraction'))
        self.neural_model.add(LeakyReLU(alpha=0.1, name='activation_abstraction'))
        
        # Capa de salida: Probabilidades por categoría
        self.neural_model.add(Dense(
            units=self.total_categories,
            activation='softmax',                    # Convierte scores a probabilidades
            name='output_probabilities'
        ))
        
        # Mostrar resumen de la arquitectura
        print(f"\n{'─'*70}")
        print(f"  ARQUITECTURA DE LA RED")
        print(f"{'─'*70}")
        self.neural_model.summary()
        print(f"{'─'*70}\n")
    
    def configure_training_strategy(self):
        """
        Configura el proceso de optimización y las estrategias de entrenamiento adaptativo.
        
        Utiliza:
        - Adam optimizer: Algoritmo de optimización adaptativo que ajusta
          los pesos de la red para minimizar el error
        - Categorical crossentropy: Función de pérdida para clasificación multiclase
        - ReduceLROnPlateau: Reduce el learning rate cuando el progreso se estanca
        - EarlyStopping: Detiene el entrenamiento si no hay mejoras significativas
        """
        print(f"[PASO 3] Configurando estrategia de optimización...")
        
        # Optimizador Adam: Ajusta pesos y sesgos de manera eficiente
        # Combina momentum y tasa de aprendizaje adaptativa
        optimization_algorithm = Adam(learning_rate=self.INITIAL_LEARNING_RATE)
        
        # Compilar el modelo con función de pérdida y métrica de evaluación
        self.neural_model.compile(
            loss='categorical_crossentropy',         # Pérdida para clasificación multiclase
            optimizer=optimization_algorithm,
            metrics=['accuracy']                     # Métrica principal: precisión
        )
        
        # ───────────────────────────────────────────────────────
        # Callback 1: Reducción adaptativa del learning rate
        # ───────────────────────────────────────────────────────
        # Cuando la red deja de mejorar, reduce el LR para hacer ajustes más finos
        self.lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',                      # Observa la pérdida de validación
            factor=self.LR_REDUCTION_FACTOR,         # Reduce LR a la mitad
            patience=self.LR_PLATEAU_PATIENCE,       # Espera 3 épocas sin mejora
            min_lr=1e-7,                             # LR mínimo permitido
            verbose=1,                               # Mostrar cuando se active
            mode='min'                               # Queremos minimizar la pérdida
        )
        
        # ───────────────────────────────────────────────────────
        # Callback 2: Detención temprana
        # ───────────────────────────────────────────────────────
        # Si no hay mejoras significativas, detiene el entrenamiento
        # y restaura los mejores pesos encontrados
        self.early_termination = EarlyStopping(
            monitor='val_loss',                      # Observa la pérdida de validación
            patience=self.EARLY_STOP_PATIENCE,       # Espera 5 épocas sin mejora
            min_delta=0.001,                         # Mejora mínima considerada significativa
            restore_best_weights=True,               # Vuelve a los mejores pesos
            verbose=1,                               # Mostrar cuando se active
            mode='min'
        )
        
        print(f"✓ Optimizador configurado: Adam (LR={self.INITIAL_LEARNING_RATE})")
        print(f"✓ Función de pérdida: Categorical Crossentropy")
        print(f"✓ Callbacks activos: ReduceLROnPlateau, EarlyStopping\n")
    
    def execute_training(self):
        """
        Ejecuta el proceso completo de entrenamiento con validación.
        
        El entrenamiento procede en épocas, donde cada época:
        1. Procesa todos los batches de entrenamiento
        2. Evalúa el rendimiento en el conjunto de validación
        3. Los callbacks ajustan parámetros o detienen si es necesario
        """
        print(f"[PASO 4] Iniciando proceso de entrenamiento...")
        print(f"\n{'═'*70}")
        print(f"  ENTRENAMIENTO EN PROGRESO")
        print(f"{'═'*70}\n")
        
        # Entrenar el modelo usando generadores
        self.training_history = self.neural_model.fit(
            self.train_data_flow,                    # Generador de datos de entrenamiento
            steps_per_epoch=len(self.train_data_flow),  # Pasos = total_muestras / batch_size
            epochs=self.TRAINING_ITERATIONS,         # Número máximo de iteraciones
            verbose=1,                               # Mostrar barra de progreso
            validation_data=self.validation_data_flow,  # Datos para evaluar generalización
            validation_steps=len(self.validation_data_flow),
            callbacks=[self.lr_scheduler, self.early_termination]  # Estrategias adaptativas
        )
        
        # Extraer la mejor métrica de validación alcanzada
        best_validation_accuracy = max(self.training_history.history['val_accuracy']) * 100
        
        print(f"\n{'═'*70}")
        print(f"  ENTRENAMIENTO COMPLETADO")
        print(f"{'═'*70}")
        print(f"🏆 Mejor precisión en validación: {best_validation_accuracy:.2f}%")
        print(f"{'═'*70}\n")
    
    def persist_model_and_metrics(self, model_filename="animal_classifier_optimized-2.h5"):
        """
        Guarda el modelo entrenado y el historial de métricas en disco.
        
        Args:
            model_filename (str): Nombre del archivo para guardar el modelo
        """
        print(f"[PASO 5] Persistiendo resultados del entrenamiento...")
        
        # Guardar el modelo completo (arquitectura + pesos + optimizador)
        self.neural_model.save(model_filename)
        print(f"✓ Modelo guardado: {model_filename}")
        
        # Guardar historial de entrenamiento (métricas por época)
        history_file = "training_history.pkl"
        with open(history_file, 'wb') as file_handler:
            pickle.dump(self.training_history.history, file_handler)
        print(f"✓ Historial de métricas guardado: {history_file}\n")
    
    def visualize_training_results(self):
        """
        Genera y guarda visualizaciones del progreso del entrenamiento.
        
        Crea dos gráficos:
        1. Evolución de la precisión (training vs validation)
        2. Evolución de la pérdida (training vs validation)
        
        Útil para diagnosticar overfitting, underfitting y convergencia.
        """
        print(f"[PASO 6] Generando visualizaciones...")
        
        # Extraer métricas del historial
        training_accuracy = self.training_history.history['accuracy']
        validation_accuracy = self.training_history.history['val_accuracy']
        training_loss = self.training_history.history['loss']
        validation_loss = self.training_history.history['val_loss']
        epoch_indices = range(1, len(training_accuracy) + 1)
        
        # Crear figura con dos subplots lado a lado
        figure, axes = plt.subplots(1, 2, figsize=(14, 6))
        figure.suptitle('Análisis de Rendimiento del Entrenamiento', fontsize=16, fontweight='bold')
        
        # ───────────────────────────────────────────────────────
        # Subplot 1: Evolución de la Precisión
        # ───────────────────────────────────────────────────────
        axes[0].plot(epoch_indices, training_accuracy, 
                     label='Precisión en Entrenamiento', 
                     color='#2E86DE', linewidth=2, marker='o', markersize=4)
        axes[0].plot(epoch_indices, validation_accuracy, 
                     label='Precisión en Validación', 
                     color='#10AC84', linewidth=2, marker='s', markersize=4)
        axes[0].set_xlabel('Época', fontsize=12)
        axes[0].set_ylabel('Precisión', fontsize=12)
        axes[0].set_title('Evolución de Precisión', fontsize=14, fontweight='bold')
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        # ───────────────────────────────────────────────────────
        # Subplot 2: Evolución de la Pérdida
        # ───────────────────────────────────────────────────────
        axes[1].plot(epoch_indices, training_loss, 
                     label='Pérdida en Entrenamiento', 
                     color='#EE5A6F', linewidth=2, marker='o', markersize=4)
        axes[1].plot(epoch_indices, validation_loss, 
                     label='Pérdida en Validación', 
                     color='#FC5C65', linewidth=2, marker='s', markersize=4)
        axes[1].set_xlabel('Época', fontsize=12)
        axes[1].set_ylabel('Pérdida', fontsize=12)
        axes[1].set_title('Evolución de Pérdida', fontsize=14, fontweight='bold')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        
        # Ajustar layout y guardar
        plt.tight_layout()
        output_filename = 'training_results.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"✓ Gráficas guardadas: {output_filename}")
        
        # Mostrar las gráficas
        plt.show()
        
        print(f"\n{'='*70}")
        print(f"  PROCESO COMPLETO FINALIZADO CON ÉXITO")
        print(f"{'='*70}\n")
    
    def run_complete_pipeline(self):
        """
        Ejecuta el pipeline completo de entrenamiento de principio a fin.
        
        Secuencia de ejecución:
        1. Configurar generadores de datos
        2. Construir arquitectura de red
        3. Configurar estrategia de optimización
        4. Ejecutar entrenamiento
        5. Guardar modelo y métricas
        6. Visualizar resultados
        """
        self.setup_data_generators()
        self.construct_network_architecture()
        self.configure_training_strategy()
        self.execute_training()
        self.persist_model_and_metrics()
        self.visualize_training_results()


def main():
    """
    Función principal que inicializa y ejecuta el sistema de entrenamiento.
    """
    # Construir ruta al dataset de manera robusta
    workspace_root = os.getcwd()
    dataset_path = os.path.join(
        workspace_root, 
        'practica_2', 
        'animals-dataset', 
        'animals-dataset'
    )
    
    # Validar que el dataset existe
    if not os.path.exists(dataset_path):
        print(f"⚠️  ERROR: No se encontró el dataset en: {dataset_path}")
        print(f"   Por favor, verifica la ruta del dataset.")
        return
    
    # Crear instancia del entrenador
    trainer = NeuralNetworkTrainer(dataset_directory=dataset_path)
    
    # Ejecutar pipeline completo
    trainer.run_complete_pipeline()


# Punto de entrada del programa
if __name__ == "__main__":
    main()