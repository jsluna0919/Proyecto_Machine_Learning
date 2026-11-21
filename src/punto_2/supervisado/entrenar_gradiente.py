
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.punto_2.preparacion import PreparadorMagic

from src.punto_2.supervisado.modelos.regresion_logistica import (
    RegresionLogisticaMagic
)

from src.punto_2.supervisado.modelos.regresion_neuronal import (
    RegresionNeuronalMagic
)

from src.punto_2.supervisado.optimizacion.gradiente import (
    GradienteDescendente
)


# Carpeta de resultados
SALIDA = (
    Path(__file__).resolve().parents[2]
    / "resultados"
    / "punto_2"
    / "gradiente"
)


def obtener_datos():

    # División inicial 80/20
    preparador = PreparadorMagic()

    X_train, _, y_train, _ = (
        preparador.dividir_datos()
    )

    # Crear validación interna utilizando
    # solamente el conjunto de entrenamiento
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
    )

    # Ajustar el escalador solo con X_fit
    escalador = StandardScaler()

    X_fit = escalador.fit_transform(X_fit)
    X_val = escalador.transform(X_val)

    return (
        X_fit,
        X_val,
        y_fit.to_numpy(),
        y_val.to_numpy()
    )


def evaluar_modelo(modelo, parametros, X_val, y_val):

    predicciones = modelo.predecir(
        X_val,
        parametros
    )

    # Cuatro métricas de clasificación
    metricas = {
        "exactitud": accuracy_score(
            y_val, predicciones
        ),

        "precision": precision_score(
            y_val, predicciones,
            zero_division=0
        ),

        "sensibilidad": recall_score(
            y_val, predicciones,
            zero_division=0
        ),

        "f1_score": f1_score(
            y_val, predicciones,
            zero_division=0
        ),
    }

    matriz = confusion_matrix(
        y_val,
        predicciones,
        labels=[0, 1]
    )

    return metricas, matriz


def graficar_convergencia(historiales):

    plt.figure(figsize=(9, 5))

    for nombre, historial in historiales.items():

        plt.plot(
            range(len(historial)),
            historial,
            label=nombre
        )

    plt.xlabel("Iteraciones")
    plt.ylabel("Pérdida")
    plt.title("Convergencia del descenso por gradiente")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        SALIDA / "convergencia_gradiente.png",
        dpi=200
    )

    plt.close()


def main():

    SALIDA.mkdir(
        parents=True,
        exist_ok=True
    )

    # Cargar datos
    X_fit, X_val, y_fit, y_val = (
        obtener_datos()
    )

    print("\n=== DIMENSIONES ===")
    print(f"Entrenamiento interno: {X_fit.shape}")
    print(f"Validación interna: {X_val.shape}")

    # Configurar modelos y tasas de aprendizaje
    configuraciones = [
        (
            "regresion_logistica",
            RegresionLogisticaMagic(),
            0.1
        ),
        (
            "regresion_neuronal",
            RegresionNeuronalMagic(),
            0.05
        ),
    ]

    resultados = []
    historiales = {}

    # Entrenar los dos modelos
    for nombre, modelo, tasa in configuraciones:

        optimizador = GradienteDescendente(
            tasa_aprendizaje=tasa,
            iteraciones=150,
            semilla=42
        )

        parametros = optimizador.optimizar(
            modelo,
            X_fit,
            y_fit
        )

        # Evaluación sobre validación interna
        metricas, matriz = evaluar_modelo(
            modelo,
            parametros,
            X_val,
            y_val
        )

        print(f"\n=== RESULTADOS: {nombre} ===")

        for metrica, valor in metricas.items():
            print(f"{metrica}: {valor:.4f}")

        print("\nMatriz de confusión:")
        print(matriz)

        # Guardar resultados
        resultados.append({
            "modelo": nombre,
            "optimizador": "gradiente_descendente",
            **metricas
        })

        historiales[nombre] = (
            optimizador.historial.copy()
        )

        # Guardar matriz
        pd.DataFrame(
            matriz,
            index=["Real 0", "Real 1"],
            columns=["Predicho 0", "Predicho 1"]
        ).to_csv(
            SALIDA / f"matriz_{nombre}.csv"
        )

        # Guardar parámetros entrenados
        np.save(
            SALIDA / f"parametros_{nombre}.npy",
            parametros
        )

    # Guardar métricas generales
    pd.DataFrame(resultados).to_csv(
        SALIDA / "metricas_gradiente.csv",
        index=False
    )

    # Guardar pérdidas de cada iteración
    pd.DataFrame(historiales).to_csv(
        SALIDA / "historial_gradiente.csv",
        index=False
    )

    # Graficar las curvas de convergencia
    graficar_convergencia(historiales)

    print(
        f"\nResultados guardados en:\n{SALIDA}"
    )


if __name__ == "__main__":
    main()
