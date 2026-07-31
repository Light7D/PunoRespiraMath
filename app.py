import cv2
import numpy as np
import streamlit as st

from motor_svd import construir_modelo_calibrado, evaluar_imagen, imagen_para_mostrar


st.set_page_config(page_title="Puno-RespiraMath | UNA Puno", page_icon="🫁", layout="wide")


@st.cache_resource(show_spinner=False)
def cargar_modelo(energia: float, sensibilidad: float):
    return construir_modelo_calibrado(
        "dataset/sanos",
        "dataset/prueba",
        energia_objetivo=energia,
        sensibilidad_objetivo=sensibilidad,
    )


st.title("🫁 Puno-RespiraMath")
st.caption("Tamizaje explicable de radiografías mediante SVD/PCA · Prototipo académico UNA Puno")
st.warning(
    "Esta herramienta no diagnostica tuberculosis. Una prioridad alta indica necesidad de revisión "
    "y de pruebas clínicas, moleculares o microbiológicas confirmatorias."
)

with st.sidebar:
    st.header("Modelo automático")
    st.write("El número de componentes y los umbrales se calculan a partir de los datos.")
    with st.expander("Configuración avanzada"):
        energia = st.slider(
            "Energía acumulada objetivo",
            0.80,
            0.99,
            0.95,
            0.01,
            help="El modelo elige automáticamente el menor k que conserva esta energía.",
        )
        sensibilidad = st.slider(
            "Sensibilidad objetivo de tamizaje",
            0.70,
            1.00,
            0.90,
            0.05,
            help="Prioriza evitar falsos negativos en el pequeño conjunto de validación disponible.",
        )

try:
    with st.spinner("Entrenando y calibrando el subespacio normal..."):
        modelo = cargar_modelo(energia, sensibilidad)
except Exception as exc:
    st.error(f"No se pudo construir el modelo: {exc}")
    st.info("Verifica que existan imágenes en dataset/sanos y dataset/prueba.")
    st.stop()

with st.sidebar:
    st.success("Modelo calibrado")
    st.metric("Componentes elegidos (k)", modelo.k)
    st.metric("Energía conservada", f"{modelo.energia:.1%}")
    st.write(f"Umbral bajo: `{modelo.tau_bajo:.4f}`")
    st.write(f"Umbral alto: `{modelo.tau_alto:.4f}`")
    st.caption(
        f"Entrenamiento: {modelo.n_entrenamiento} sanas · Validación: "
        f"{modelo.n_validacion_sanos} sanas y {modelo.n_validacion_anomalos} anómalas"
    )

st.subheader("Validación interna del prototipo")
m = modelo.metricas_validacion
c1, c2, c3 = st.columns(3)
c1.metric("Sensibilidad", f"{m['sensibilidad']:.1%}")
c2.metric("Especificidad", f"{m['especificidad']:.1%}")
c3.metric("AUC", f"{m['auc']:.3f}")
st.caption(
    "Métricas exploratorias calculadas con pocas imágenes locales. No equivalen a validación clínica "
    "ni garantizan el rendimiento en pacientes nuevos."
)

st.divider()
st.subheader("Evaluación de una radiografía")
archivo = st.file_uploader(
    "Seleccione o arrastre una radiografía de tórax",
    type=["jpg", "jpeg", "png"],
)

if archivo is None:
    st.info("Carga una imagen para obtener su mapa de anomalías y prioridad de revisión.")
    st.stop()

try:
    resultado = evaluar_imagen(archivo.getvalue(), modelo)
except Exception as exc:
    st.error(f"No se pudo procesar la imagen: {exc}")
    st.stop()

if resultado["avisos_calidad"]:
    st.warning("Control de calidad: " + " ".join(resultado["avisos_calidad"]))

col1, col2, col3 = st.columns(3)
col1.image(resultado["original"], caption="1. Radiografía recibida", clamp=True, use_container_width=True)
col2.image(
    imagen_para_mostrar(resultado["reconstruida"], modelo.mascara),
    caption="2. Reconstrucción desde el subespacio normal",
    use_container_width=True,
)
residuo = resultado["residuo"]
residuo_8 = (residuo / max(float(np.max(residuo)), 1e-8) * 255).astype(np.uint8)
mapa = cv2.cvtColor(cv2.applyColorMap(residuo_8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
mapa[~modelo.mascara] = 0
col3.image(mapa, caption="3. Mapa de anomalías dentro de la región pulmonar", use_container_width=True)

st.divider()
score = resultado["score"]
prioridad = resultado["prioridad"]
r1, r2 = st.columns([1, 2])
r1.metric("Puntuación S(X)", f"{score:.4f}")
with r2:
    if prioridad == "Alta":
        st.error("🚨 Prioridad alta: radiografía atípica; requiere evaluación adicional.")
    elif prioridad == "Intermedia":
        st.warning("⚠️ Prioridad intermedia: se recomienda revisión profesional.")
    else:
        st.success("Prioridad baja según este modelo, sin descartar evaluación clínica.")

st.latex(r"S(X)=\frac{\|M\odot(X-\hat X)\|_F}{\sqrt{\sum_{i,j}M(i,j)}}")
st.caption(
    "La puntuación mide diferencia respecto del subespacio aprendido; puede aumentar por neumonía, "
    "fibrosis, mala inspiración, artefactos u otras alteraciones, no únicamente por tuberculosis."
)
