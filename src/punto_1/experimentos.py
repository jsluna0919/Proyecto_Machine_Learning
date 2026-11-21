from dataclasses import dataclass
from itertools import product

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.punto_1.modelos.arbol_decision import ModeloArbolDecision
from src.punto_1.modelos.red_neuronal import ModeloRedNeuronal
from src.punto_1.modelos.regresion_logistica import ModeloRegresionLogistica
from src.punto_1.modelos.svm import ModeloSVM
from src.punto_1.seleccion.correlacion import SelectorCorrelacion
from src.punto_1.seleccion.pca import SelectorPCA


@dataclass(frozen=True)
class Experimento:
    identificador: str
    modelo: str
    seleccion: str
    parametros: dict


def generar_experimentos() -> list[Experimento]:
    """4 algoritmos × 2 técnicas × 4 hiperparámetros = 32 modelos."""
    opciones = {
        "logistica": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
        "red_neuronal": [
            {"hidden_layer_sizes": capas} for capas in ((16,), (32,), (64,), (32, 16))
        ],
        "svm": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
        "arbol": [{"max_depth": profundidad} for profundidad in (3, 5, 8, 12)],
    }
    experimentos = []
    for modelo, seleccion in product(opciones, ("correlacion", "pca")):
        for indice, parametros in enumerate(opciones[modelo], 1):
            experimentos.append(Experimento(
                identificador=f"{modelo}_{seleccion}_{indice:02d}",
                modelo=modelo,
                seleccion=seleccion,
                parametros=parametros,
            ))
    return experimentos


def construir_pipeline(experimento: Experimento) -> Pipeline:
    constructores = {
        "logistica": ModeloRegresionLogistica,
        "red_neuronal": ModeloRedNeuronal,
        "svm": ModeloSVM,
        "arbol": ModeloArbolDecision,
    }
    if experimento.seleccion == "correlacion":
        seleccion = SelectorCorrelacion(k=20)
    elif experimento.seleccion == "pca":
        seleccion = SelectorPCA.crear(varianza_explicada=0.95)
    else:
        raise ValueError(f"Selección desconocida: {experimento.seleccion}")
    modelo = constructores[experimento.modelo].crear(**experimento.parametros)
    return Pipeline([
        ("escalado", StandardScaler()),
        ("seleccion", seleccion),
        ("modelo", modelo),
    ])
