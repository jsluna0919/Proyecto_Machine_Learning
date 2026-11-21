
"""Validación cruzada de los 32 experimentos del punto 1.

Ejemplos de ejecución:

    # Ejecutar dos modelos de prueba
    python -m src.punto_1.validacion_cruzada --max-modelos 2

    # Ejecutar todos los modelos
    python -m src.punto_1.validacion_cruzada --modelo todos

    # Ejecutar por algoritmo
    python -m src.punto_1.validacion_cruzada --modelo logistica
    python -m src.punto_1.validacion_cruzada --modelo arbol
    python -m src.punto_1.validacion_cruzada --modelo svm
    python -m src.punto_1.validacion_cruzada --modelo red_neuronal
"""

import argparse
import json

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from src.config import RESULTADOS_PUNTO_1, SEMILLA
from src.comun.metricas import calcular_metricas

from src.punto_1.datos import cargar_datos, dividir_datos
from src.punto_1.experimentos import (
    construir_pipeline,
    generar_experimentos,
)


# ==========================================
# 1. CONFIGURACIÓN DE LAS COLUMNAS DEL CSV
# ==========================================

COLUMNAS_ORDEN = [
    "modelo_id",
    "modelo",
    "seleccion",
    "parametros",
    "cv_folds",
    "f1_macro",
    "f1_macro_std",
    "sensibilidad_fraude",
    "precision_macro",
    "sensibilidad_macro",
    "especificidad_macro",
    "exactitud",
    "fowlkes_mallows_macro",
    "verdaderos_positivos",
    "falsos_negativos",
    "falsos_positivos",
    "verdaderos_negativos",
]


# ==========================================
# 2. EVALUACIÓN CON VALIDACIÓN CRUZADA
# ==========================================

def evaluar_cv(experimento, X_train, y_train, cv):
    """
    Entrena y evalúa un modelo mediante validación
    cruzada estratificada.

    No utiliza el conjunto de prueba final.
    """

    resultados_folds = []

    # Matriz de confusión acumulada
    matriz_total = np.zeros((2, 2), dtype=np.int64)

    # Recorrer cada partición
    for numero, (idx_fit, idx_val) in enumerate(
        cv.split(X_train, y_train), 1
    ):

        # Separar entrenamiento y validación
        X_fit = X_train.iloc[idx_fit]
        X_val = X_train.iloc[idx_val]

        y_fit = y_train.iloc[idx_fit]
        y_val = y_train.iloc[idx_val]

        # Construir el pipeline del experimento
        # Incluye escalado, selección y modelo
        pipeline = construir_pipeline(experimento)

        # Entrenar solamente con datos del fold
        pipeline.fit(X_fit, y_fit)

        # Generar predicciones
        predicciones = pipeline.predict(X_val)

        # Calcular métricas
        metricas = calcular_metricas(
            y_val,
            predicciones
        )

        resultados_folds.append(metricas)

        # Acumular matriz de confusión
        matriz = confusion_matrix(
            y_val,
            predicciones,
            labels=[0, 1]
        )

        matriz_total += matriz

        # Mostrar avance
        print(
            f"    Fold {numero}/{cv.n_splits} | "
            f"F1 macro: {metricas['f1_macro']:.4f} | "
            f"Sensibilidad fraude: "
            f"{metricas['sensibilidad_fraude']:.4f}",
            flush=True
        )

    # ======================================
    # Promediar métricas de los folds
    # ======================================

    fila = {
        "modelo_id": experimento.identificador,
        "modelo": experimento.modelo,
        "seleccion": experimento.seleccion,
        "parametros": json.dumps(
            experimento.parametros,
            ensure_ascii=False
        ),
        "cv_folds": cv.n_splits,
    }

    for metrica in resultados_folds[0]:

        valores = [
            resultado[metrica]
            for resultado in resultados_folds
        ]

        fila[metrica] = float(np.mean(valores))

    # Desviación estándar del F1 macro
    fila["f1_macro_std"] = float(
        np.std(
            [r["f1_macro"] for r in resultados_folds],
            ddof=1
        )
    )

    # Extraer valores de matriz de confusión
    tn, fp, fn, tp = matriz_total.ravel()

    fila.update({
        "verdaderos_positivos": int(tp),
        "falsos_negativos": int(fn),
        "falsos_positivos": int(fp),
        "verdaderos_negativos": int(tn),
    })

    return fila


# ==========================================
# 3. GUARDAR RESULTADOS
# ==========================================

def guardar_resultados(resultados, destino):
    """
    Guarda los resultados en CSV y evita duplicados.
    """

    tabla = pd.DataFrame(resultados)

    tabla = tabla.drop_duplicates(
        subset=["modelo_id", "cv_folds"],
        keep="last"
    )

    tabla = tabla.reindex(columns=COLUMNAS_ORDEN)

    tabla.to_csv(
        destino,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"    Resultado guardado: {destino}",
        flush=True
    )


# ==========================================
# 4. MOSTRAR RANKING DE MODELOS
# ==========================================

def mostrar_ranking(resultados, cv_folds):
    """
    Muestra los cinco modelos con mejor F1 macro.
    """

    tabla = pd.DataFrame(resultados)

    if tabla.empty:
        print("\nNo hay resultados para mostrar.")
        return

    # Filtrar el número de folds utilizado
    tabla = tabla[
        tabla["cv_folds"] == cv_folds
    ].copy()

    if tabla.empty:
        print("\nNo hay resultados para esta validación.")
        return

    # Ordenar por rendimiento
    tabla = tabla.sort_values(
        by=["f1_macro", "sensibilidad_fraude"],
        ascending=[False, False]
    )

    columnas = [
        "modelo_id",
        "f1_macro",
        "precision_macro",
        "sensibilidad_macro",
        "sensibilidad_fraude",
        "exactitud",
    ]

    print("\n" + "=" * 60)
    print("TOP 5 MODELOS - VALIDACIÓN CRUZADA")
    print("=" * 60)

    print(
        tabla[columnas]
        .head(5)
        .round(4)
        .to_string(index=False)
    )


# ==========================================
# 5. FUNCIÓN PRINCIPAL
# ==========================================

def main():

    # --------------------------------------
    # Configurar argumentos
    # --------------------------------------

    parser = argparse.ArgumentParser(
        description="Entrenamiento y validación de 32 modelos"
    )

    parser.add_argument(
        "--modelo",
        default="todos",
        choices=[
            "todos",
            "logistica",
            "red_neuronal",
            "svm",
            "arbol",
        ],
        help="Algoritmo que se desea entrenar"
    )

    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Número de folds de validación cruzada"
    )

    parser.add_argument(
        "--max-modelos",
        type=int,
        default=None,
        help="Limitar la cantidad de modelos para pruebas"
    )

    args = parser.parse_args()

    # Validaciones
    if args.cv < 2:
        parser.error("--cv debe ser al menos 2")

    if (
        args.max_modelos is not None
        and args.max_modelos < 1
    ):
        parser.error("--max-modelos debe ser positivo")

    # --------------------------------------
    # Cargar los datos
    # --------------------------------------

    print("\n=== CARGANDO BASE DE DATOS ===", flush=True)

    X, y = cargar_datos()

    # Mantener el conjunto de prueba reservado
    X_train, _, y_train, _ = dividir_datos(X, y)

    print(
        f"Entrenamiento: {X_train.shape}",
        flush=True
    )

    print(
        "El conjunto de prueba permanece reservado.",
        flush=True
    )

    # --------------------------------------
    # Obtener experimentos
    # --------------------------------------

    experimentos = generar_experimentos()

    # Filtrar por algoritmo
    if args.modelo != "todos":

        experimentos = [
            e for e in experimentos
            if e.modelo == args.modelo
        ]

    # Limitar experimentos si se solicita
    if args.max_modelos is not None:

        experimentos = experimentos[
            :args.max_modelos
        ]

    print(
        f"\nConfiguraciones seleccionadas: "
        f"{len(experimentos)}",
        flush=True
    )

    if not experimentos:
        print("No se encontraron experimentos.")
        return

    # --------------------------------------
    # Configurar validación cruzada
    # --------------------------------------

    cv = StratifiedKFold(
        n_splits=args.cv,
        shuffle=True,
        random_state=SEMILLA
    )

    # --------------------------------------
    # Crear carpeta de resultados
    # --------------------------------------

    RESULTADOS_PUNTO_1.mkdir(
        parents=True,
        exist_ok=True
    )

    destino = (
        RESULTADOS_PUNTO_1 /
        "validacion_cruzada.csv"
    )

    # --------------------------------------
    # Recuperar resultados existentes
    # --------------------------------------

    if destino.exists():

        resultados = pd.read_csv(
            destino
        ).to_dict("records")

        print(
            f"\nResultados anteriores cargados: "
            f"{len(resultados)}",
            flush=True
        )

    else:
        resultados = []

    # Identificar modelos ya completados
    completados = {
        (r["modelo_id"], int(r["cv_folds"]))
        for r in resultados
    }

    # --------------------------------------
    # Entrenar modelos
    # --------------------------------------

    print("\n=== INICIANDO EXPERIMENTOS ===")

    for indice, experimento in enumerate(
        experimentos, 1
    ):

        clave = (
            experimento.identificador,
            args.cv
        )

        # Evitar repetir experimentos
        if clave in completados:

            print(
                f"\n[{indice}/{len(experimentos)}] "
                f"Ya evaluado: {clave[0]}",
                flush=True
            )

            continue

        print(
            f"\n[{indice}/{len(experimentos)}] "
            f"Entrenando: {experimento.identificador}",
            flush=True
        )

        # Ejecutar validación cruzada
        fila = evaluar_cv(
            experimento,
            X_train,
            y_train,
            cv
        )

        # Almacenar resultado en memoria
        resultados.append(fila)

        # Marcar experimento como completado
        completados.add(clave)

        # Guardar CSV después de cada modelo
        guardar_resultados(
            resultados,
            destino
        )

        print(
            f"    Finalizado: {experimento.identificador}",
            flush=True
        )

    # --------------------------------------
    # Mostrar resultados finales
    # --------------------------------------

    print("\n=== EXPERIMENTOS FINALIZADOS ===")

    if resultados:

        mostrar_ranking(
            resultados,
            args.cv
        )

        print(
            f"\nResultados disponibles en:\n{destino}"
        )

    else:
        print("No se generaron resultados.")


# ==========================================
# 6. PUNTO DE ENTRADA
# ==========================================

if __name__ == "__main__":
    main()