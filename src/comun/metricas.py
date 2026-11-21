import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def calcular_metricas(y_real, y_predicho) -> dict[str, float]:
    """Métricas macro para clasificación binaria 0/1.

    La exactitud estándar coincide con el promedio de exactitudes
    one-vs-rest cuando el problema tiene dos clases.
    """
    cm = confusion_matrix(y_real, y_predicho, labels=[0, 1])
    especificidades = []
    fm_por_clase = []
    precisiones = precision_score(
        y_real, y_predicho, average=None, labels=[0, 1], zero_division=0
    )
    sensibilidades = recall_score(
        y_real, y_predicho, average=None, labels=[0, 1], zero_division=0
    )
    for i in range(2):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        especificidades.append(tn / (tn + fp) if tn + fp else 0.0)
        fm_por_clase.append(np.sqrt(precisiones[i] * sensibilidades[i]))

    return {
        "exactitud": float(accuracy_score(y_real, y_predicho)),
        "precision_macro": float(precision_score(y_real, y_predicho, average="macro", labels=[0, 1], zero_division=0)),
        "sensibilidad_macro": float(recall_score(y_real, y_predicho, average="macro", labels=[0, 1], zero_division=0)),
        "especificidad_macro": float(np.mean(especificidades)),
        "f1_macro": float(f1_score(y_real, y_predicho, average="macro", labels=[0, 1], zero_division=0)),
        "fowlkes_mallows_macro": float(np.mean(fm_por_clase)),
        "sensibilidad_fraude": float(sensibilidades[1]),
    }
