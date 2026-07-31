"""Motor matematico de Puno-RespiraMath.

Implementa el flujo descrito en el articulo: control de calidad, registro,
normalizacion dentro de una mascara pulmonar aproximada, PCA mediante SVD,
seleccion automatica de k por energia y calibracion de umbrales con validacion.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union

import cv2
import numpy as np


IMG_SIZE = (256, 256)
ImageSource = Union[str, Path, bytes, bytearray]


@dataclass
class ModeloSVD:
    media: np.ndarray
    base: np.ndarray
    valores_singulares: np.ndarray
    mascara: np.ndarray
    referencia: np.ndarray
    k: int
    energia: float
    tau_bajo: float
    tau_alto: float
    metricas_validacion: dict
    n_entrenamiento: int
    n_validacion_sanos: int
    n_validacion_anomalos: int


def listar_imagenes(carpeta: Union[str, Path]) -> list[str]:
    carpeta = Path(carpeta)
    if not carpeta.exists():
        return []
    extensiones = {".png", ".jpg", ".jpeg"}
    return sorted(str(p) for p in carpeta.iterdir() if p.suffix.lower() in extensiones)


def _leer_grises(origen: ImageSource) -> np.ndarray:
    if isinstance(origen, (bytes, bytearray)):
        datos = np.frombuffer(origen, dtype=np.uint8)
        img = cv2.imdecode(datos, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(str(origen), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("No se pudo leer la radiografia.")
    return cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)


def crear_mascara_pulmonar() -> np.ndarray:
    """Mascara anatomica aproximada para el prototipo (no es segmentacion clinica)."""
    mascara = np.zeros(IMG_SIZE, dtype=np.uint8)
    cv2.ellipse(mascara, (91, 132), (48, 92), 2, 0, 360, 1, -1)
    cv2.ellipse(mascara, (165, 132), (48, 92), -2, 0, 360, 1, -1)
    mascara[:35, :] = 0
    mascara[225:, :] = 0
    return mascara.astype(bool)


def evaluar_calidad(img_8bit: np.ndarray) -> tuple[bool, list[str]]:
    avisos = []
    media = float(np.mean(img_8bit))
    contraste = float(np.std(img_8bit))
    recorte = float(np.mean((img_8bit <= 2) | (img_8bit >= 253)))
    if media < 35:
        avisos.append("La imagen parece subexpuesta.")
    if media > 220:
        avisos.append("La imagen parece sobreexpuesta.")
    if contraste < 22:
        avisos.append("El contraste es demasiado bajo.")
    if recorte > 0.35:
        avisos.append("Hay demasiados pixeles saturados o recortados.")
    return len(avisos) == 0, avisos


def _registrar(img: np.ndarray, referencia: np.ndarray) -> np.ndarray:
    """Registro euclidiano suave; si ECC falla, conserva la imagen original."""
    movimiento = np.eye(2, 3, dtype=np.float32)
    criterios = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
    try:
        cv2.findTransformECC(
            referencia.astype(np.float32) / 255.0,
            img.astype(np.float32) / 255.0,
            movimiento,
            cv2.MOTION_EUCLIDEAN,
            criterios,
            None,
            3,
        )
        return cv2.warpAffine(
            img,
            movimiento,
            IMG_SIZE,
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
    except cv2.error:
        return img


def _normalizar(img_8bit: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img_8bit).astype(np.float64) / 255.0
    valores = img[mascara]
    media = float(np.mean(valores))
    desviacion = float(np.std(valores))
    normalizada = np.zeros_like(img)
    normalizada[mascara] = (valores - media) / (desviacion + 1e-8)
    return normalizada


def preprocesar_imagen(
    origen: ImageSource, referencia: np.ndarray, mascara: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    original = _leer_grises(origen)
    _, avisos = evaluar_calidad(original)
    registrada = _registrar(original, referencia)
    return original.astype(np.float64) / 255.0, _normalizar(registrada, mascara), avisos


def _puntuar_vector(vector: np.ndarray, media: np.ndarray, base: np.ndarray, n_mask: int):
    centrado = vector - media
    reconstruido = media + base @ (base.T @ centrado)
    score = float(np.linalg.norm(vector - reconstruido) / np.sqrt(n_mask))
    return score, reconstruido


def _metricas(scores_normales: np.ndarray, scores_anomalos: np.ndarray, umbral: float) -> dict:
    sensibilidad = float(np.mean(scores_anomalos >= umbral)) if len(scores_anomalos) else np.nan
    especificidad = float(np.mean(scores_normales < umbral)) if len(scores_normales) else np.nan
    if len(scores_normales) and len(scores_anomalos):
        comparaciones = scores_anomalos[:, None] - scores_normales[None, :]
        auc = float(np.mean(comparaciones > 0) + 0.5 * np.mean(comparaciones == 0))
    else:
        auc = np.nan
    return {"sensibilidad": sensibilidad, "especificidad": especificidad, "auc": auc}


def _calibrar_umbrales(
    scores_normales: np.ndarray, scores_anomalos: np.ndarray, sensibilidad_objetivo: float
) -> tuple[float, float, dict]:
    if not len(scores_normales):
        raise ValueError("Se necesitan radiografias normales de validacion.")
    if not len(scores_anomalos):
        tau1 = float(np.quantile(scores_normales, 0.90))
        tau2 = float(np.quantile(scores_normales, 0.99))
        return tau1, max(tau1, tau2), _metricas(scores_normales, scores_anomalos, tau1)

    valores = np.unique(np.concatenate([scores_normales, scores_anomalos]))
    # Los puntos medios evitan depender de si una observacion cae exactamente
    # sobre el umbral.
    candidatos = np.concatenate(
        ([np.nextafter(valores[0], -np.inf)], (valores[:-1] + valores[1:]) / 2, [np.nextafter(valores[-1], np.inf)])
    )
    evaluaciones = []
    for t in candidatos:
        m = _metricas(scores_normales, scores_anomalos, float(t))
        evaluaciones.append((float(t), m["sensibilidad"], m["especificidad"]))

    aptos = [e for e in evaluaciones if e[1] >= sensibilidad_objetivo]
    tau1 = max(aptos, key=lambda e: (e[2], e[0]))[0] if aptos else float(np.min(candidatos))
    # La frontera alta privilegia especificidad: una alerta roja debe ser más
    # exigente que la frontera inicial de tamizaje.
    alta_especificidad = [e for e in evaluaciones if e[2] >= 0.95 and e[0] > tau1]
    tau2 = (
        max(alta_especificidad, key=lambda e: (e[1], -e[0]))[0]
        if alta_especificidad
        else float(np.nextafter(tau1, np.inf))
    )
    metricas = _metricas(scores_normales, scores_anomalos, tau1)
    metricas["sensibilidad_objetivo"] = sensibilidad_objetivo
    return tau1, tau2, metricas


def construir_modelo_calibrado(
    carpeta_sanos: Union[str, Path],
    carpeta_anomalos: Union[str, Path],
    energia_objetivo: float = 0.95,
    sensibilidad_objetivo: float = 0.90,
    semilla: int = 42,
) -> ModeloSVD:
    sanos = listar_imagenes(carpeta_sanos)
    anomalos = listar_imagenes(carpeta_anomalos)
    if len(sanos) < 6:
        raise ValueError("Se requieren al menos 6 radiografias sanas.")

    rng = np.random.default_rng(semilla)
    indices = rng.permutation(len(sanos))
    n_validacion = max(2, int(round(len(sanos) * 0.30)))
    validacion_sanos = [sanos[i] for i in indices[:n_validacion]]
    entrenamiento = [sanos[i] for i in indices[n_validacion:]]

    crudas = [_leer_grises(r) for r in entrenamiento]
    referencia = np.median(np.stack(crudas), axis=0).astype(np.uint8)
    mascara = crear_mascara_pulmonar()
    procesadas = [_normalizar(_registrar(img, referencia), mascara) for img in crudas]
    X = np.column_stack([img.flatten() for img in procesadas])
    media = np.mean(X, axis=1)
    A = X - media[:, None]
    U, S, _ = np.linalg.svd(A, full_matrices=False)
    energia_acumulada = np.cumsum(S**2) / max(float(np.sum(S**2)), 1e-12)
    k = min(int(np.searchsorted(energia_acumulada, energia_objetivo) + 1), len(entrenamiento) - 1)
    base = U[:, :k]

    def puntuar_rutas(rutas: Iterable[str]) -> np.ndarray:
        salida = []
        for ruta in rutas:
            _, proc, _ = preprocesar_imagen(ruta, referencia, mascara)
            score, _ = _puntuar_vector(proc.flatten(), media, base, int(mascara.sum()))
            salida.append(score)
        return np.asarray(salida, dtype=float)

    scores_normales = puntuar_rutas(validacion_sanos)
    scores_anomalos = puntuar_rutas(anomalos)
    tau1, tau2, metricas = _calibrar_umbrales(
        scores_normales, scores_anomalos, sensibilidad_objetivo
    )
    metricas["scores_normales"] = scores_normales.tolist()
    metricas["scores_anomalos"] = scores_anomalos.tolist()

    return ModeloSVD(
        media=media,
        base=base,
        valores_singulares=S,
        mascara=mascara,
        referencia=referencia,
        k=k,
        energia=float(energia_acumulada[k - 1]),
        tau_bajo=tau1,
        tau_alto=tau2,
        metricas_validacion=metricas,
        n_entrenamiento=len(entrenamiento),
        n_validacion_sanos=len(validacion_sanos),
        n_validacion_anomalos=len(anomalos),
    )


def evaluar_imagen(origen: ImageSource, modelo: ModeloSVD) -> dict:
    original, procesada, avisos = preprocesar_imagen(origen, modelo.referencia, modelo.mascara)
    score, reconstruido_v = _puntuar_vector(
        procesada.flatten(), modelo.media, modelo.base, int(modelo.mascara.sum())
    )
    reconstruida = reconstruido_v.reshape(IMG_SIZE)
    residuo = np.abs(procesada - reconstruida) * modelo.mascara
    if score < modelo.tau_bajo:
        prioridad = "Baja"
    elif score < modelo.tau_alto:
        prioridad = "Intermedia"
    else:
        prioridad = "Alta"
    return {
        "original": original,
        "procesada": procesada,
        "reconstruida": reconstruida,
        "residuo": residuo,
        "score": score,
        "prioridad": prioridad,
        "avisos_calidad": avisos,
    }


def imagen_para_mostrar(matriz: np.ndarray, mascara: np.ndarray | None = None) -> np.ndarray:
    valores = matriz[mascara] if mascara is not None else matriz.ravel()
    bajo, alto = np.percentile(valores, [1, 99])
    normalizada = np.clip((matriz - bajo) / max(alto - bajo, 1e-8), 0, 1)
    return (normalizada * 255).astype(np.uint8)


# Compatibilidad con la primera version del proyecto.
def cargar_y_vectorizar_imagen(ruta_imagen):
    img = _leer_grises(ruta_imagen)
    return img.astype(np.float64) / 255.0
