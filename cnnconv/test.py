"""
Sistema de Clasificación de Animales usando CNN
Implementa la clasificación de imágenes de animales utilizando
una red neuronal convolucional previamente entrenada.
"""

import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import requests
from pathlib import Path


class AnimalClassifier:
    """
    Clasificador de imágenes de animales basado en CNN.
    
    Atributos:
        IMAGE_WIDTH (int): Ancho de entrada requerido por la CNN
        IMAGE_HEIGHT (int): Alto de entrada requerido por la CNN
        CATEGORIES (list): Lista de categorías de animales
    """
    
    # Dimensiones de entrada para la red neuronal
    IMAGE_WIDTH = 128
    IMAGE_HEIGHT = 128
    
    # Categorías de clasificación en el orden correcto
    CATEGORIES = ["catarina", "gato", "hormiga", "perro", "tortuga"]
    
    def __init__(self, model_path="animal_classifier_optimized-2.h5"):
        """
        Inicializa el clasificador cargando el modelo entrenado.
        
        Args:
            model_path (str): Ruta al archivo del modelo .h5
        """
        print(f"[INFO] Cargando modelo desde: {model_path}")
        self.neural_network = load_model(model_path)
        print("[INFO] Modelo cargado exitosamente")
    
    def fetch_image_from_url(self, image_url, destination="downloaded_temp.jpg"):
        """
        Descarga una imagen desde una URL especificada.
        
        Args:
            image_url (str): URL de la imagen a descargar
            destination (str): Ruta donde guardar la imagen descargada
            
        Returns:
            str o None: Ruta del archivo descargado o None si falla
        """
        try:
            # Realizar petición HTTP para obtener la imagen
            http_response = requests.get(image_url, stream=True, timeout=10)
            http_response.raise_for_status()
            
            # Escribir los datos de la imagen en el archivo
            with open(destination, 'wb') as file_handler:
                for data_chunk in http_response.iter_content(chunk_size=8192):
                    file_handler.write(data_chunk)
            
            print(f"✓ Descarga completada: {destination}")
            return destination
            
        except requests.exceptions.RequestException as error:
            print(f"✗ Fallo en la descarga: {error}")
            return None
    
    def prepare_image_for_network(self, image_path):
        """
        Preprocesa una imagen para que sea compatible con la CNN.
        
        Proceso:
        1. Lee la imagen desde el disco
        2. Convierte el espacio de color BGR a RGB
        3. Redimensiona manteniendo el aspect ratio
        4. Añade padding para alcanzar las dimensiones requeridas
        5. Normaliza los valores de píxeles al rango [0, 1]
        6. Añade dimensión batch para la predicción
        
        Args:
            image_path (str): Ruta de la imagen a procesar
            
        Returns:
            np.ndarray: Imagen procesada lista para la CNN
            
        Raises:
            ValueError: Si la imagen no se puede cargar
        """
        # Leer la imagen del sistema de archivos
        raw_image = cv2.imread(image_path)
        
        # Validar que la imagen se cargó correctamente
        if raw_image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # Convertir de BGR (formato OpenCV) a RGB (formato estándar)
        rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        
        # Calcular el factor de escala manteniendo el aspect ratio
        original_height, original_width = rgb_image.shape[:2]
        scaling_factor = min(
            self.IMAGE_WIDTH / original_width,
            self.IMAGE_HEIGHT / original_height
        )
        
        # Calcular nuevas dimensiones escaladas
        scaled_width = int(original_width * scaling_factor)
        scaled_height = int(original_height * scaling_factor)
        
        # Redimensionar la imagen con el nuevo tamaño
        resized_image = cv2.resize(rgb_image, (scaled_width, scaled_height))
        
        # Crear canvas con padding (relleno negro)
        canvas = np.zeros(
            (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3),
            dtype=np.uint8
        )
        
        # Calcular offsets para centrar la imagen
        horizontal_offset = (self.IMAGE_WIDTH - scaled_width) // 2
        vertical_offset = (self.IMAGE_HEIGHT - scaled_height) // 2
        
        # Colocar la imagen redimensionada en el centro del canvas
        canvas[
            vertical_offset:vertical_offset + scaled_height,
            horizontal_offset:horizontal_offset + scaled_width
        ] = resized_image
        
        # Normalizar valores de píxeles de [0, 255] a [0, 1]
        normalized_canvas = canvas.astype("float32") / 255.0
        
        # Expandir dimensiones para crear batch: (altura, ancho, canales) -> (1, altura, ancho, canales)
        batched_image = np.expand_dims(normalized_canvas, axis=0)
        
        return batched_image
    
    def classify_image(self, image_path):
        """
        Clasifica una imagen y retorna los resultados de la predicción.
        
        Args:
            image_path (str): Ruta de la imagen a clasificar
            
        Returns:
            tuple: (categoría_predicha, vector_probabilidades, confianza_porcentaje)
        """
        # Preprocesar la imagen para la red neuronal
        processed_input = self.prepare_image_for_network(image_path)
        
        # Ejecutar la inferencia del modelo
        probability_vector = self.neural_network.predict(processed_input)
        
        # Obtener el índice de la clase con mayor probabilidad
        predicted_index = np.argmax(probability_vector)
        
        # Obtener la categoría correspondiente al índice
        predicted_category = self.CATEGORIES[predicted_index]
        
        # Calcular la confianza en porcentaje
        confidence_percentage = probability_vector[0][predicted_index] * 100
        
        return predicted_category, probability_vector, confidence_percentage
    
    def visualize_prediction(self, image_path, category, confidence):
        """
        Muestra la imagen con la predicción superpuesta.
        
        Args:
            image_path (str): Ruta de la imagen a visualizar
            category (str): Categoría predicha
            confidence (float): Confianza de la predicción en porcentaje
        """
        # Cargar y convertir la imagen para visualización
        display_image = cv2.imread(image_path)
        display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        
        # Crear la figura de matplotlib
        plt.imshow(display_image)
        
        # Añadir texto con la predicción
        text_x_position = display_image.shape[1] - 10
        text_y_position = 30
        
        plt.text(
            text_x_position, text_y_position, category,
            fontsize=14, color='white',
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
        )
        
        # Configurar visualización y mostrar
        plt.axis("off")
        plt.show()


def main():
    """
    Función principal que ejecuta el flujo de clasificación.
    """
    # Instanciar el clasificador
    classifier = AnimalClassifier()
    
    # Rutas de prueba disponibles (comentadas)
    # test_paths = [
    #     "cnnconv\\test-images\\catarina5.jpg",
    #     "cnnconv\\test-images\\gato6.png",
    #     "cnnconv\\test-images\\hormiga1.jpeg",
    #     "cnnconv\\test-images\\perro4.jpg",
    # ]
    
    # Ruta de la imagen a clasificar
    test_image_path = "cnnconv\\test-images\\dog16.jpg"
    
    print(f"\n[INFO] Procesando imagen: {test_image_path}")
    
    # Realizar la clasificación
    category, probabilities, confidence = classifier.classify_image(test_image_path)
    
    # Mostrar resultados en consola
    print(f"\n{'='*50}")
    print(f"Categoría detectada: {category}")
    print(f"Vector de probabilidades: {probabilities}")
    print(f"Confianza: {confidence:.2f}%")
    print(f"{'='*50}\n")
    
    # Visualizar la imagen con la predicción
    classifier.visualize_prediction(test_image_path, category, confidence)


# Punto de entrada del programa
if __name__ == "__main__":
    main()