
"""Evaluación final del apartado 2(i) del taller.

Modelos seleccionados mediante validación interna:
    - Regresión logística + Algoritmo genético
    - Regresión neuronal + PSO

Ejecutar:
    python -m src.punto_2.evaluacion_final
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.punto_2.preparacion import PreparadorMagic

from src.punto_2.supervisado.modelos.regresion_logistica import (
    RegresionLogisticaMagic
)

from src.punto_2.supervisado.modelos.regresion_neuronal import (
    RegresionNeuronalMagic
)

from src.punto_2.supervisado.optimizacion.genetico import (
    AlgoritmoGenetico
)

from src.punto_2.supervisado.optimizacion.pso import (
    OptimizadorPSO
)


# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

RAIZ = Path(__file__).resolve().parents[2]

SALIDA = (
    RAIZ
    / "resultados"
    / "punto_2"
    / "evaluacion_final"
)

SEMILLA = 42


# ==========================================
# 2. PREPARAR DATOS DEFINITIVOS
# ==========================================

def preparar_datos_finales():

    preparador = PreparadorMagic(
        test_size=0.20,
        random_state=42
    )

    X_train, X_test, y_train, y_test = (
        preparador.dividir_datos()
    )

    # Ajustar solamente con entrenamiento
    escalador = StandardScaler()

    X_train = escalador.fit_transform(X_train)
    X_test = escalador.transform(X_test)

    return (
        X_train,
        X_test,
        y_train.to_numpy(),
        y_test.to_numpy(),
        escalador
    )


# ==========================================
# 3. CONFIGURAR MODELOS SELECCIONADOS
# ==========================================

def obtener_experimentos():

    return [
        (
            "regresion_logistica",
            "genetico",
            RegresionLogisticaMagic(),
            AlgoritmoGenetico(
                poblacion=20,
                generaciones=40,
                semilla=SEMILLA
            )
        ),
        (
            "regresion_neuronal",
            "pso",
            RegresionNeuronalMagic(),
            OptimizadorPSO(
                particulas=20,
                iteraciones=40,
                semilla=SEMILLA
            )
        )
    ]


# ==========================================
# 4. EVALUAR MODELO
# ==========================================

def evaluar(modelo, parametros, X_test, y_test):

    predicciones = modelo.predecir(
        X_test,
        parametros
    )

    metricas = {
        "exactitud": accuracy_score(
            y_test, predicciones
        ),
        "precision": precision_score(
            y_test, predicciones,
            zero_division=0
        ),
        "sensibilidad": recall_score(
            y_test, predicciones,
            zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predicciones,
            zero_division=0
        )
    }

    matriz = confusion_matrix(
        y_test,
        predicciones,
        labels=[0, 1]
    )

    return metricas, matriz


# ==========================================
# 5. GUARDAR MATRIZ DE CONFUSIÓN
# ==========================================

def guardar_matriz(matriz, nombre):

    # Guardar datos de la matriz
    pd.DataFrame(
        matriz,
        index=["Real hadron", "Real gamma"],
        columns=["Predicho hadron", "Predicho gamma"]
    ).to_csv(
        SALIDA / f"matriz_{nombre}.csv"
    )

    # Guardar gráfica
    fig, ax = plt.subplots(figsize=(6, 5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=[
            "Hadron (0)",
            "Gamma (1)"
        ]
    )

    display.plot(
        ax=ax,
        values_format="d",
        colorbar=False
    )

    ax.set_title(
        f"Matriz de confusion - {nombre}"
    )

    fig.tight_layout()

    fig.savefig(
        SALIDA / f"matriz_{nombre}.png",
        dpi=250
    )

    plt.close(fig)


# ==========================================
# 6. FUNCIÓN PRINCIPAL
# ==========================================

def main():

    SALIDA.mkdir(
        parents=True,
        exist_ok=True
    )

    archivo = SALIDA / "metricas_finales.csv"

    print("\n=== CARGANDO DATOS MAGIC ===")

    X_train, X_test, y_train, y_test, escalador = (
        preparar_datos_finales()
    )

    print(f"Entrenamiento: {X_train.shape}")
    print(f"Prueba: {X_test.shape}")

    # Conservar el escalador para inferencias
    joblib.dump(
        escalador,
        SALIDA / "escalador_final.joblib"
    )

    # Recuperar resultados si existen
    if archivo.exists():
        registros = pd.read_csv(
            archivo
        ).to_dict("records")
    else:
        registros = []

    completados = {
        (r["modelo"], r["optimizador"])
        for r in registros
    }

    experimentos = obtener_experimentos()

    print("\n=== EVALUACIÓN FINAL ===")

    for nombre, metodo, modelo, optimizador in experimentos:

        if (nombre, metodo) in completados:
            print(f"Ya evaluado: {nombre} + {metodo}")
            continue

        print(
            f"\nEntrenando: {nombre} + {metodo}",
            flush=True
        )

        inicio = perf_counter()

        # Entrenar con todos los registros disponibles
        parametros = optimizador.optimizar(
            modelo,
            X_train,
            y_train
        )

        tiempo = perf_counter() - inicio

        # Evaluar sobre el conjunto de prueba
        metricas, matriz = evaluar(
            modelo,
            parametros,
            X_test,
            y_test
        )

        perdida = modelo.perdida(
            parametros,
            X_train,
            y_train
        )

        tn, fp, fn, tp = matriz.ravel()

        # Registrar los resultados
        registro = {
            "modelo": nombre,
            "optimizador": metodo,
            "semilla": SEMILLA,
            "perdida_entrenamiento": perdida,
            "tiempo_segundos": tiempo,
            **metricas,
            "verdaderos_negativos": int(tn),
            "falsos_positivos": int(fp),
            "falsos_negativos": int(fn),
            "verdaderos_positivos": int(tp)
        }

        registros.append(registro)

        # Guardar parámetros
        np.save(
            SALIDA / f"parametros_{nombre}_{metodo}.npy",
            parametros
        )

        # Guardar matriz
        guardar_matriz(
            matriz,
            f"{nombre}_{metodo}"
        )

        # Guardar métricas
        pd.DataFrame(registros).to_csv(
            archivo,
            index=False,
            encoding="utf-8-sig"
        )

        print("\nMétricas de prueba:")

        for metrica, valor in metricas.items():
            print(f"  {metrica}: {valor:.4f}")

        print("\nMatriz de confusión:")
        print(matriz)

        print(f"\nTiempo: {tiempo:.2f} segundos")

    # ======================================
    # RESUMEN FINAL
    # ======================================

    tabla = pd.read_csv(archivo)

    columnas = [
        "modelo",
        "optimizador",
        "exactitud",
        "precision",
        "sensibilidad",
        "f1_score",
        "falsos_positivos",
        "falsos_negativos"
    ]

    print("\n=== RESUMEN FINAL ===")

    print(
        tabla[columnas]
        .round(4)
        .to_string(index=False)
    )

    print(
        f"\nArchivos guardados en:\n{SALIDA}"
    )

    print(
        "\nLa evaluación final no debe utilizarse "
        "para ajustar los hiperparámetros."
    )


if __name__ == "__main__":
    main()
