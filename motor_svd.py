import os
import cv2
import numpy as np

# Dimensiones estándar para las matrices de imagen
IMG_SIZE = (256, 256)

def cargar_y_vectorizar_imagen(ruta_imagen):
    """
    Lee la imagen en escala de grises, aplica ecualización de contraste 
    (CLAHE) para estandarizar iluminaciones y la redimensiona a 256x256.
    """
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
    
    # Redimensión estándar
    img_resized = cv2.resize(img, IMG_SIZE)
    
    # Normalización de Contraste (CLAHE) - Clave para SVD
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_equalized = clahe.apply(img_resized)
    
    # Normalización matricial a rango [0, 1]
    matrix = img_equalized.astype(np.float64) / 255.0
    return matrix

def construir_subespacio_sano(carpeta_sanos, k_componentes=15):
    """
    Aplica SVD sobre el conjunto de entrenamiento de pulmones sanos
    para extraer los k componentes principales (Base Uk).
    """
    archivos = [os.path.join(carpeta_sanos, f) for f in os.listdir(carpeta_sanos) 
                if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(archivos) == 0:
        raise FileNotFoundError("No se encontraron imágenes en la carpeta 'dataset/sanos/'.")

    # Vectorización de imágenes para construir la matriz de datos X
    vectores = []
    for ruta in archivos:
        img_matrix = cargar_y_vectorizar_imagen(ruta)
        vectores.append(img_matrix.flatten()) # Convierte la matriz 256x256 en vector de 65536
    
    # Matriz de datos X de tamaño (m_pixeles x n_muestras)
    X = np.column_stack(vectores)
    
    # Descomposición en Valores Singulares (SVD)
    # X = U * S * V^T
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Truncamiento a los k componentes principales del subespacio sano
    Uk = U[:, :k_componentes]
    return Uk

def evaluar_anomalia_svd(matriz_test, Uk):
    """
    Proyecta la matriz del paciente sobre el subespacio Uk y calcula
    el error de reconstrucción relativo usando la Norma de Frobenius.
    """
    v_test = matriz_test.flatten()
    
    # Operador de Proyección Ortogonal: P = Uk * Uk^T * v_test
    proyeccion_v = Uk @ (Uk.T @ v_test)
    
    # Reconstrucción de la matriz proyectada
    matriz_proyectada = proyeccion_v.reshape(IMG_SIZE)
    
    # Error Relativo en Norma de Frobenius (Independiente de la escala)
    norma_original = np.linalg.norm(matriz_test, 'fro')
    delta = np.linalg.norm(matriz_test - matriz_proyectada, 'fro') / norma_original
    
    return delta, matriz_proyectada
    
    return delta, matriz_proyectada