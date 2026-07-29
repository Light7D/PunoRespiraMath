import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from motor_svd import cargar_y_vectorizar_imagen, construir_subespacio_sano, evaluar_anomalia_svd

# Configuración de la página en Streamlit
st.set_page_config(page_title="Puno-RespiraMath | UNA Puno", layout="wide")

st.title("🫁 Puno-RespiraMath")
st.caption("Sistema de Tamizaje Pulmonar Mediante Descomposición en Valores Singulares (SVD)")
st.markdown("---")

# Barra lateral para parámetros matemáticos
st.sidebar.header("⚙️ Parámetros del Modelo")
k_comp = st.sidebar.slider("Número de Componentes Singulares (k)", min_value=5, max_value=50, value=15, step=5)
umbral_tau = st.sidebar.number_input("Umbral de Alerta (Tau)", value=12.5, step=0.5)

carpeta_sanos = "dataset/sanos"

# Construcción o carga del subespacio sano Uk
try:
    with st.spinner("Construyendo subespacio ortonormal Uk mediante SVD..."):
        Uk = construir_subespacio_sano(carpeta_sanos, k_componentes=k_comp)
    st.sidebar.success(f"Subespacio Uk construido correctamente con k={k_comp}.")
except Exception as e:
    st.sidebar.error(f"Error al cargar dataset: {e}")
    st.info("💡 **Instrucción:** Agrega algunas imágenes .jpg/.png en la carpeta `dataset/sanos/` para iniciar.")
    st.stop()

# Carga de la imagen del paciente
st.subheader("📋 Evaluación de Paciente (Tamizaje en Tiempo Real)")
archivo_subido = st.file_uploader("Seleccione o arrastre una radiografía de tórax (JPG/PNG)", type=["jpg", "png", "jpeg"])

if archivo_subido is not None:
    # Guardar archivo temporal
    ruta_temp = "temp_paciente.png"
    with open(ruta_temp, "wb") as f:
        f.write(archivo_subido.getbuffer())
    
    # Procesar imagen
    A_test = cargar_y_vectorizar_imagen(ruta_temp)
    
    # Evaluar anomaía vía SVD y Norma de Frobenius
    delta, A_reconstruida = evaluar_anomalia_svd(A_test, Uk)
    
    # Despliegue de Resultados Visuales
   # Despliegue de Resultados Visuales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Convertimos la matriz normalizada (0-1) a imagen de 8 bits (0-255) para Streamlit
        st.image((A_test * 255).astype(np.uint8), caption="1. Radiografía Entrante (A_test)", use_container_width=True)
    
    with col2:
        st.image((A_reconstruida * 255).astype(np.uint8), caption="2. Proyección sobre Subespacio Sano (P)", use_container_width=True)
        
    with col3:
        # Residuo visual (Diferencia de geometrías) en mapa de calor usando OpenCV
        residuo = np.abs(A_test - A_reconstruida)
        residuo_norm = (residuo / np.max(residuo) * 255).astype(np.uint8) if np.max(residuo) > 0 else residuo.astype(np.uint8)
        heatmap = cv2.applyColorMap(residuo_norm, cv2.COLORMAP_JET)
        st.image(heatmap, caption="3. Mapa de Residuo / Invarianza", use_container_width=True)
    st.markdown("---")
    
    # Despliegue de la Métrica Matematica y Diagnóstico de Alerta
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        st.metric(label="Métrica de Desviación (Norma de Frobenius δ)", value=f"{delta:.4f}")
        st.latex(r"\delta = || A_{test} - P ||_F")
        
    with m_col2:
        if delta > umbral_tau:
            st.error("🚨 **ALERTA DETECTADA: Se observan patrones anómalos fuera del subespacio sano.**")
            st.warning("Prioridad: Derivar a Baciloscopía / Examen Clínico Preferencial.")
        else:
            st.success("✅ **PATRÓN NORMAL: La geometría matricial coincide con el subespacio de salud.**")
            st.info("Sin anomalías geométricas detectadas por SVD.")