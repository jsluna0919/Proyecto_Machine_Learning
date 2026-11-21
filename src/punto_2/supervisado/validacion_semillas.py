
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

from src.punto_2.supervisado.optimizacion.gradiente import (
    GradienteDescendente
)
from src.punto_2.supervisado.optimizacion.genetico import (
    AlgoritmoGenetico
)
from src.punto_2.supervisado.optimizacion.pso import (
    OptimizadorPSO
)
from src.punto_2.supervisado.optimizacion.simulated_annealing import (
    SimulatedAnnealing
)


SALIDA = (
    Path(__file__).resolve().parents[2]
    / "resultados"
    / "punto_2"
    / "comparacion_semillas"
)

SEMILLAS = [42, 123, 2026]

MODELOS = {
    "regresion_logistica": RegresionLogisticaMagic,
    "regresion_neuronal": RegresionNeuronalMagic
}

METODOS = [
    "Gradiente",
    "Genetico",
    "PSO",
    "SA"
]


def crear_optimizador(metodo, nombre_modelo, semilla):

    if metodo == "Gradiente":

        tasa = (
            0.1 if nombre_modelo == "regresion_logistica"
            else 0.05
        )

        return GradienteDescendente(
            tasa_aprendizaje=tasa,
            iteraciones=150,
            semilla=semilla
        )

    if metodo == "Genetico":

        return AlgoritmoGenetico(
            poblacion=20,
            generaciones=40,
            semilla=semilla
        )

    if metodo == "PSO":

        return OptimizadorPSO(
            particulas=20,
            iteraciones=40,
            semilla=semilla
        )

    if metodo == "SA":

        return SimulatedAnnealing(
            evaluaciones=800,
            semilla=semilla
        )

    raise ValueError(f"Optimizador desconocido: {metodo}")


def ejecutar_experimentos():

    X_fit, X_val, y_fit, y_val = obtener_datos()

    SALIDA.mkdir(parents=True, exist_ok=True)

    archivo = SALIDA / "resultados_semillas.csv"

    if archivo.exists():
        registros = pd.read_csv(archivo).to_dict("records")
    else:
        registros = []

    completados = {
        (
            r["modelo"],
            r["optimizador"],
            int(r["semilla"])
        )
        for r in registros
    }

    for nombre, clase_modelo in MODELOS.items():

        for metodo in METODOS:

            for semilla in SEMILLAS:

                clave = (nombre, metodo, semilla)

                if clave in completados:
                    print(f"Ya evaluado: {clave}")
                    continue

                print(
                    f"\n=== {nombre} | {metodo} | "
                    f"Semilla {semilla} ===",
                    flush=True
                )

                modelo = clase_modelo()

                optimizador = crear_optimizador(
                    metodo,
                    nombre,
                    semilla
                )

                inicio = perf_counter()

                parametros = optimizador.optimizar(
                    modelo,
                    X_fit,
                    y_fit
                )

                tiempo = perf_counter() - inicio

                metricas, matriz = evaluar_modelo(
                    modelo,
                    parametros,
                    X_val,
                    y_val
                )

                perdida = modelo.perdida(
                    parametros,
                    X_fit,
                    y_fit
                )

                tn, fp, fn, tp = matriz.ravel()

                registro = {
                    "modelo": nombre,
                    "optimizador": metodo,
                    "semilla": semilla,
                    "perdida_entrenamiento": perdida,
                    "tiempo_segundos": tiempo,
                    "verdaderos_positivos": int(tp),
                    "falsos_negativos": int(fn),
                    "falsos_positivos": int(fp),
                    "verdaderos_negativos": int(tn),
                    **metricas
                }

                registros.append(registro)
                completados.add(clave)

                # Guardar después de cada ejecución
                pd.DataFrame(registros).to_csv(
                    archivo,
                    index=False,
                    encoding="utf-8-sig"
                )

                print(
                    f"F1-score: {metricas['f1_score']:.4f}"
                )

    return pd.DataFrame(registros)


def analizar_resultados(datos):

    resumen = datos.groupby(
        ["modelo", "optimizador"],
        as_index=False
    ).agg(
        f1_promedio=("f1_score", "mean"),
        f1_std=("f1_score", "std"),
        exactitud_promedio=("exactitud", "mean"),
        precision_promedio=("precision", "mean"),
        sensibilidad_promedio=("sensibilidad", "mean"),
        perdida_promedio=("perdida_entrenamiento", "mean"),
        tiempo_promedio=("tiempo_segundos", "mean"),
        repeticiones=("semilla", "nunique")
    )

    resumen = resumen.sort_values(
        ["modelo", "f1_promedio"],
        ascending=[True, False]
    )

    resumen.to_csv(
        SALIDA / "resumen_semillas.csv",
        index=False,
        encoding="utf-8-sig"
    )

    for nombre in MODELOS:

        print(f"\n=== {nombre.upper()} ===")

        subset = resumen[
            resumen["modelo"] == nombre
        ]

        print(
            subset[
                [
                    "optimizador",
                    "f1_promedio",
                    "f1_std",
                    "precision_promedio",
                    "sensibilidad_promedio",
                    "perdida_promedio"
                ]
            ].round(4).to_string(index=False)
        )

    return resumen


def graficar_resultados(resumen):

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(METODOS))
    ancho = 0.35

    for indice, nombre in enumerate(MODELOS):

        subset = resumen[
            resumen["modelo"] == nombre
        ].set_index("optimizador")

        subset = subset.reindex(METODOS)

        valores = subset["f1_promedio"].to_numpy()
        desviaciones = subset["f1_std"].fillna(0).to_numpy()

        desplazamiento = (
            indice - 0.5
        ) * ancho

        ax.bar(
            x + desplazamiento,
            valores,
            width=ancho,
            yerr=desviaciones,
            capsize=4,
            label=nombre.replace("_", " ").title()
        )

    ax.set_xticks(x)
    ax.set_xticklabels(METODOS)

    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-score promedio")

    ax.set_title(
        "Comparación de optimizadores con distintas semillas"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()

    fig.savefig(
        SALIDA / "comparacion_semillas.png",
        dpi=250
    )

    plt.close(fig)


def main():

    datos = ejecutar_experimentos()

    resumen = analizar_resultados(datos)

    graficar_resultados(resumen)

    print(
        f"\nResultados guardados en:\n{SALIDA}"
    )


if __name__ == "__main__":
    main()
