
"""Evaluación final de los mejores modelos del punto 1.

Ejecutar desde la raíz del proyecto:

    python -m src.punto_1.evaluacion_final

Los modelos se seleccionan utilizando los resultados de
validación cruzada. El conjunto de prueba se utiliza
únicamente para la evaluación final.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
)

from src.config import RESULTADOS_PUNTO_1
from src.comun.metricas import calcular_metricas

from src.punto_1.datos import cargar_datos, dividir_datos
from src.punto_1.experimentos import (
    construir_pipeline,
    generar_experimentos,
)


# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

FOLDS_CV = 3

FAMILIAS = (
    "logistica",
    "red_neuronal",
    "svm",
    "arbol",
)


# ==========================================
# 2. SELECCIÓN DE FINALISTAS
# ==========================================

def seleccionar_finalistas(ruta_cv: Path):
    """
    Selecciona los cinco mejores modelos según F1 macro
    y el mejor representante de cada algoritmo.

    La selección se realiza exclusivamente con los
    resultados de validación cruzada.
    """

    if not ruta_cv.is_file():
        raise FileNotFoundError(
            f"No encuentro la validación cruzada: {ruta_cv}"
        )

    # Cargar resultados
    resultados = pd.read_csv(ruta_cv)

    # Utilizar experimentos evaluados con 3 folds
    resultados = resultados.loc[
        resultados["cv_folds"] == FOLDS_CV
    ].copy()

    # Evitar duplicados
    resultados = resultados.drop_duplicates(
        subset="modelo_id",
        keep="last"
    )

    if resultados.empty:
        raise ValueError(
            f"No existen modelos evaluados con {FOLDS_CV} folds."
        )

    # Ordenar por F1 macro
    ranking = resultados.sort_values(
        ["f1_macro", "sensibilidad_fraude"],
        ascending=[False, False]
    )

    # Seleccionar los cinco mejores
    top5 = ranking.head(5)["modelo_id"].tolist()

    # Seleccionar el mejor de cada familia
    representantes = []

    for familia in FAMILIAS:

        subset = ranking.loc[
            ranking["modelo"] == familia
        ]

        if subset.empty:
            raise ValueError(
                f"Faltan resultados de la familia: {familia}"
            )

        representantes.append(
            subset.iloc[0]["modelo_id"]
        )

    # Eliminar repetidos conservando el orden
    finalistas = list(
        dict.fromkeys(top5 + representantes)
    )

    return (
        finalistas,
        top5,
        ranking.set_index("modelo_id")
    )


# ==========================================
# 3. ENTRENAMIENTO Y EVALUACIÓN FINAL
# ==========================================

def evaluar_finalista(
    experimento,
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Entrena el modelo con todos los datos de
    entrenamiento y evalúa sobre el conjunto de prueba.
    """

    # Construir el pipeline
    pipeline = construir_pipeline(experimento)

    # Entrenar el modelo definitivo
    pipeline.fit(X_train, y_train)

    # Realizar predicciones
    predicciones = pipeline.predict(X_test)

    # Calcular las métricas solicitadas
    metricas = calcular_metricas(
        y_test,
        predicciones
    )

    # Matriz de confusión
    matriz = confusion_matrix(
        y_test,
        predicciones,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matriz.ravel()

    # Construir registro de resultados
    resultado = {
        "modelo_id": experimento.identificador,
        "modelo": experimento.modelo,
        "seleccion": experimento.seleccion,

        **metricas,

        # Métricas específicas de fraude
        "precision_fraude": float(
            precision_score(
                y_test,
                predicciones,
                pos_label=1,
                zero_division=0
            )
        ),

        "f1_fraude": float(
            f1_score(
                y_test,
                predicciones,
                pos_label=1,
                zero_division=0
            )
        ),

        # Matriz de confusión
        "verdaderos_positivos": int(tp),
        "falsos_negativos": int(fn),
        "falsos_positivos": int(fp),
        "verdaderos_negativos": int(tn),
    }

    return resultado, matriz


# ==========================================
# 4. GRAFICAR MATRIZ DE CONFUSIÓN
# ==========================================

def guardar_matriz(
    matriz,
    modelo_id: str,
    carpeta: Path
):
    """
    Genera y guarda la matriz de confusión
    correspondiente al modelo evaluado.
    """

    figura, eje = plt.subplots(
        figsize=(5.7, 4.7)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=[
            "Normal (0)",
            "Fraude (1)"
        ]
    )

    display.plot(
        ax=eje,
        values_format="d",
        colorbar=False
    )

    eje.set_title(
        f"Prueba final: {modelo_id}"
    )

    figura.tight_layout()

    # Guardar imagen PNG
    figura.savefig(
        carpeta / f"matriz_prueba_{modelo_id}.png",
        dpi=200
    )

    plt.close(figura)


# ==========================================
# 5. FUNCIÓN PRINCIPAL
# ==========================================

def main():

    # --------------------------------------
    # Cargar ranking de validación cruzada
    # --------------------------------------

    origen_cv = (
        RESULTADOS_PUNTO_1 /
        "validacion_cruzada.csv"
    )

    finalistas, top5, ranking_cv = (
        seleccionar_finalistas(origen_cv)
    )

    # Obtener los experimentos configurados
    disponibles = {
        experimento.identificador: experimento
        for experimento in generar_experimentos()
    }

    # Comprobar que existan todos los modelos
    faltantes = [
        nombre
        for nombre in finalistas
        if nombre not in disponibles
    ]

    if faltantes:
        raise ValueError(
            f"Modelos desconocidos en el CSV: {faltantes}"
        )

    # --------------------------------------
    # Preparar carpeta de resultados
    # --------------------------------------

    salida = (
        RESULTADOS_PUNTO_1 /
        "evaluacion_final"
    )

    salida.mkdir(
        parents=True,
        exist_ok=True
    )

    archivo_resultados = (
        salida /
        "metricas_prueba_final.csv"
    )

    # --------------------------------------
    # Mostrar modelos seleccionados
    # --------------------------------------

    print(
        "\n=== MODELOS FINALISTAS ===",
        flush=True
    )

    for nombre in finalistas:

        f1_cv = ranking_cv.loc[
            nombre,
            "f1_macro"
        ]

        print(
            f"{nombre} | F1 macro CV: {f1_cv:.4f}",
            flush=True
        )

    # --------------------------------------
    # Cargar y dividir datos
    # --------------------------------------

    print(
        "\n=== CARGANDO BASE DE DATOS ===",
        flush=True
    )

    X, y = cargar_datos()

    # Mantener la misma división inicial
    X_train, X_test, y_train, y_test = (
        dividir_datos(X, y)
    )

    print(
        f"Entrenamiento: {X_train.shape}",
        flush=True
    )

    print(
        f"Prueba: {X_test.shape}",
        flush=True
    )

    # --------------------------------------
    # Recuperar evaluaciones anteriores
    # --------------------------------------

    if archivo_resultados.exists():

        registros = pd.read_csv(
            archivo_resultados
        ).to_dict("records")

    else:
        registros = []

    # Modelos ya evaluados
    completados = {
        registro["modelo_id"]
        for registro in registros
    }

    # --------------------------------------
    # Evaluar modelos finalistas
    # --------------------------------------

    print(
        "\n=== INICIANDO EVALUACIÓN FINAL ===",
        flush=True
    )

    for indice, nombre in enumerate(
        finalistas, 1
    ):

        # Evitar evaluar nuevamente
        if nombre in completados:

            print(
                f"[{indice}/{len(finalistas)}] "
                f"Ya evaluado: {nombre}",
                flush=True
            )

            continue

        print(
            f"\n[{indice}/{len(finalistas)}] "
            f"Entrenando: {nombre}",
            flush=True
        )

        # Entrenar y evaluar
        registro, matriz = evaluar_finalista(
            disponibles[nombre],
            X_train,
            X_test,
            y_train,
            y_test
        )

        # Registrar F1 obtenido durante CV
        registro["f1_macro_cv"] = float(
            ranking_cv.loc[
                nombre,
                "f1_macro"
            ]
        )

        # Identificar el grupo del modelo
        registro["grupo_comparacion"] = (
            "Top 5"
            if nombre in top5
            else "Representante de familia"
        )

        # Guardar matriz de confusión
        guardar_matriz(
            matriz,
            nombre,
            salida
        )

        # Almacenar métricas
        registros.append(registro)

        # Guardar resultados después de cada modelo
        pd.DataFrame(registros).to_csv(
            archivo_resultados,
            index=False,
            encoding="utf-8-sig"
        )

        completados.add(nombre)

        # Mostrar resultado
        print(
            f"  F1 macro prueba: "
            f"{registro['f1_macro']:.4f}"
        )

        print(
            f"  Sensibilidad fraude: "
            f"{registro['sensibilidad_fraude']:.4f}"
        )

        print(
            f"  Falsos positivos: "
            f"{registro['falsos_positivos']}"
        )

        print(
            f"  Falsos negativos: "
            f"{registro['falsos_negativos']}"
        )

        print(
            f"  Resultado guardado: "
            f"{archivo_resultados}",
            flush=True
        )

    # --------------------------------------
    # Mostrar resumen final
    # --------------------------------------

    tabla = pd.read_csv(
        archivo_resultados
    )

    # Mostrar solamente los finalistas
    tabla = tabla[
        tabla["modelo_id"].isin(finalistas)
    ].copy()

    # Mantener el orden de validación cruzada
    tabla["orden_cv"] = tabla["modelo_id"].map(
        {
            modelo: indice
            for indice, modelo in enumerate(finalistas)
        }
    )

    tabla = tabla.sort_values("orden_cv")

    print(
        "\n=== RESULTADOS DE LA EVALUACIÓN FINAL ==="
    )

    columnas = [
        "modelo_id",
        "f1_macro_cv",
        "f1_macro",
        "precision_fraude",
        "sensibilidad_fraude",
        "falsos_positivos",
        "falsos_negativos"
    ]

    print(
        tabla[columnas]
        .round(4)
        .to_string(index=False)
    )

    print(
        f"\nResultados guardados en:\n{salida}"
    )

    print(
        "\nIMPORTANTE: El conjunto de prueba "
        "no debe utilizarse para ajustar hiperparámetros."
    )


# ==========================================
# 6. PUNTO DE ENTRADA
# ==========================================

if __name__ == "__main__":
    main()