
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from src.punto_2.supervisado.entrenar_gradiente import (
    obtener_datos,
    evaluar_modelo
)

from src.punto_2.supervisado.modelos.regresion_logistica import (
    RegresionLogisticaMagic
)

from src.punto_2.supervisado.modelos.regresion_neuronal import (
    RegresionNeuronalMagic
)

from src.punto_2.supervisado.optimizacion.pso import (
    OptimizadorPSO
)


SALIDA = (
    Path(__file__).resolve().parents[2]
    / "resultados"
    / "punto_2"
    / "pso"
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prueba",
        action="store_true",
        help="Ejecutar una prueba rápida"
    )

    args = parser.parse_args()

    # Cargar los mismos datos que GD y GA
    X_fit, X_val, y_fit, y_val = obtener_datos()

    salida = SALIDA

    if args.prueba:

        X_fit, _, y_fit, _ = train_test_split(
            X_fit,
            y_fit,
            train_size=1000,
            random_state=42,
            stratify=y_fit
        )

        salida = SALIDA / "prueba"

        particulas = 8
        iteraciones = 3

    else:
        particulas = 20
        iteraciones = 40

    salida.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n=== DIMENSIONES ===")
    print(f"Entrenamiento: {X_fit.shape}")
    print(f"Validación: {X_val.shape}")

    modelos = [
        (
            "regresion_logistica",
            RegresionLogisticaMagic()
        ),
        (
            "regresion_neuronal",
            RegresionNeuronalMagic()
        )
    ]

    resultados = []
    historiales = {}

    for nombre, modelo in modelos:

        optimizador = OptimizadorPSO(
            particulas=particulas,
            iteraciones=iteraciones,
            semilla=42
        )

        inicio = perf_counter()

        parametros = optimizador.optimizar(
            modelo,
            X_fit,
            y_fit
        )

        tiempo = perf_counter() - inicio

        # Evaluación sobre validación interna
        metricas, matriz = evaluar_modelo(
            modelo,
            parametros,
            X_val,
            y_val
        )

        perdida_final = modelo.perdida(
            parametros,
            X_fit,
            y_fit
        )

        print(f"\n=== RESULTADOS: {nombre} ===")

        print(
            f"Pérdida final: {perdida_final:.6f}"
        )

        for metrica, valor in metricas.items():
            print(f"{metrica}: {valor:.4f}")

        print("\nMatriz de confusión:")
        print(matriz)

        print(
            f"Tiempo: {tiempo:.2f} segundos"
        )

        # Guardar métricas
        resultados.append({
            "modelo": nombre,
            "optimizador": "pso",
            "perdida_entrenamiento": perdida_final,
            "evaluaciones_objetivo": optimizador.evaluaciones,
            "tiempo_segundos": tiempo,
            **metricas
        })

        historiales[nombre] = (
            optimizador.historial.copy()
        )

        # Guardar matrices de confusión
        pd.DataFrame(
            matriz,
            index=["Real 0", "Real 1"],
            columns=["Predicho 0", "Predicho 1"]
        ).to_csv(
            salida / f"matriz_{nombre}.csv"
        )

        # Guardar parámetros
        np.save(
            salida / f"parametros_{nombre}.npy",
            parametros
        )

        # Guardar métricas después de cada modelo
        pd.DataFrame(resultados).to_csv(
            salida / "metricas_pso.csv",
            index=False
        )

    # Guardar historial de convergencia
    pd.DataFrame(historiales).to_csv(
        salida / "historial_pso.csv",
        index=False
    )

    # Graficar convergencia
    plt.figure(figsize=(9, 5))

    for nombre, historial in historiales.items():
        plt.plot(
            range(1, len(historial) + 1),
            historial,
            label=nombre
        )

    plt.xlabel("Iteraciones")
    plt.ylabel("Mejor pérdida acumulada")
    plt.title("Convergencia del optimizador PSO")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        salida / "convergencia_pso.png",
        dpi=200
    )

    plt.close()

    print(
        f"\nResultados guardados en:\n{salida}"
    )


if __name__ == "__main__":
    main()
