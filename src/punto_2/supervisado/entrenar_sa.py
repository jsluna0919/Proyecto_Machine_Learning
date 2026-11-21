
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

from src.punto_2.supervisado.optimizacion.simulated_annealing import (
    SimulatedAnnealing
)


SALIDA = (
    Path(__file__).resolve().parents[2]
    / "resultados"
    / "punto_2"
    / "simulated_annealing"
)


def main():

    parser = argparse.ArgumentParser(
        description="Entrenar MAGIC mediante Simulated Annealing"
    )

    parser.add_argument(
        "--prueba",
        action="store_true",
        help="Ejecutar una prueba rápida"
    )

    args = parser.parse_args()

    # Mismos datos que gradiente, GA y PSO
    X_fit, X_val, y_fit, y_val = obtener_datos()

    salida = SALIDA

    if args.prueba:

        # Subconjunto estratificado solo
        # para comprobar el programa
        X_fit, _, y_fit, _ = train_test_split(
            X_fit,
            y_fit,
            train_size=1000,
            random_state=42,
            stratify=y_fit
        )

        salida = SALIDA / "prueba"
        evaluaciones = 25

    else:
        evaluaciones = 800

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

    # ----------------------------------
    # Entrenar ambos modelos
    # ----------------------------------

    for nombre, modelo in modelos:

        optimizador = SimulatedAnnealing(
            evaluaciones=evaluaciones,
            temperatura_inicial=1.0,
            temperatura_final=0.001,
            sigma_mutacion=0.3,
            semilla=42
        )

        inicio = perf_counter()

        parametros = optimizador.optimizar(
            modelo,
            X_fit,
            y_fit
        )

        tiempo = perf_counter() - inicio

        # Evaluar en validación interna
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
            "optimizador": "simulated_annealing",
            "perdida_entrenamiento": perdida_final,
            "evaluaciones_objetivo": optimizador.evaluaciones,
            "tiempo_segundos": tiempo,
            "movimientos_aceptados": optimizador.aceptados,
            **metricas
        })

        historiales[nombre] = (
            optimizador.historial.copy()
        )

        # Guardar matriz de confusión
        pd.DataFrame(
            matriz,
            index=["Real 0", "Real 1"],
            columns=["Predicho 0", "Predicho 1"]
        ).to_csv(
            salida / f"matriz_{nombre}.csv"
        )

        # Guardar parámetros entrenados
        np.save(
            salida / f"parametros_{nombre}.npy",
            parametros
        )

        # Guardar resultados parciales
        pd.DataFrame(resultados).to_csv(
            salida / "metricas_sa.csv",
            index=False
        )

    # ----------------------------------
    # Guardar historial de convergencia
    # ----------------------------------

    pd.DataFrame(historiales).to_csv(
        salida / "historial_sa.csv",
        index=False
    )

    # ----------------------------------
    # Graficar convergencia
    # ----------------------------------

    plt.figure(figsize=(9, 5))

    for nombre, historial in historiales.items():

        plt.plot(
            range(1, len(historial) + 1),
            historial,
            label=nombre
        )

    plt.xlabel(
        "Evaluaciones de la función objetivo"
    )

    plt.ylabel("Mejor pérdida acumulada")

    plt.title(
        "Convergencia de Simulated Annealing"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        salida / "convergencia_sa.png",
        dpi=200
    )

    plt.close()

    print(
        f"\nResultados guardados en:\n{salida}"
    )


if __name__ == "__main__":
    main()
