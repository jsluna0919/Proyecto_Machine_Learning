
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

RESULTADOS = (
    Path(__file__).resolve().parents[2]
    / "resultados"
    / "punto_2"
)

SALIDA = RESULTADOS / "comparacion"

ARCHIVOS = {
    "Gradiente": "gradiente/metricas_gradiente.csv",
    "Genético": "genetico/metricas_genetico.csv",
    "PSO": "pso/metricas_pso.csv",
    "SA": "simulated_annealing/metricas_sa.csv",
}

MODELOS = [
    "regresion_logistica",
    "regresion_neuronal"
]

METODOS = list(ARCHIVOS.keys())

METRICAS = [
    "exactitud",
    "precision",
    "sensibilidad",
    "f1_score",
]

# ==========================================
# 2. CARGAR RESULTADOS
# ==========================================

def cargar_resultados():

    resultados = []

    # El gradiente no guardó la pérdida final
    # en metricas_gradiente.csv. La recuperamos
    # desde su historial.
    ruta_historial = (
        RESULTADOS
        / "gradiente"
        / "historial_gradiente.csv"
    )

    historial_gradiente = pd.read_csv(
        ruta_historial
    )

    for metodo, archivo in ARCHIVOS.items():

        ruta = RESULTADOS / archivo

        if not ruta.exists():
            raise FileNotFoundError(
                f"No existe el archivo: {ruta}"
            )

        tabla = pd.read_csv(ruta)

        for _, fila in tabla.iterrows():

            modelo = fila["modelo"]

            if metodo == "Gradiente":

                perdida = float(
                    historial_gradiente[modelo]
                    .dropna()
                    .iloc[-1]
                )

            else:
                perdida = float(
                    fila["perdida_entrenamiento"]
                )

            resultados.append({
                "modelo": modelo,
                "optimizador": metodo,
                "perdida": perdida,
                "exactitud": float(
                    fila["exactitud"]
                ),
                "precision": float(
                    fila["precision"]
                ),
                "sensibilidad": float(
                    fila["sensibilidad"]
                ),
                "f1_score": float(
                    fila["f1_score"]
                ),
                "tiempo_segundos": fila.get(
                    "tiempo_segundos",
                    np.nan
                )
            })

    return pd.DataFrame(resultados)


# ==========================================
# 3. GRAFICAR MÉTRICAS
# ==========================================

def graficar_comparacion(tabla, metrica):

    posiciones = np.arange(len(METODOS))
    ancho = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, modelo in enumerate(MODELOS):

        datos = tabla[
            tabla["modelo"] == modelo
        ].set_index("optimizador")

        valores = (
            datos.reindex(METODOS)[metrica]
            .to_numpy()
        )

        desplazamiento = (
            i - (len(MODELOS) - 1) / 2
        ) * ancho

        barras = ax.bar(
            posiciones + desplazamiento,
            valores,
            width=ancho,
            label=modelo.replace("_", " ").title()
        )

        ax.bar_label(
            barras,
            fmt="%.4f",
            padding=3,
            fontsize=8
        )

    ax.set_xticks(posiciones)
    ax.set_xticklabels(METODOS)

    ax.set_ylabel(metrica.replace("_", " ").title())
    ax.set_title(
        f"Comparación de optimizadores - {metrica}"
    )

    if metrica != "perdida":
        ax.set_ylim(0, 1.08)

    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()

    fig.savefig(
        SALIDA / f"comparacion_{metrica}.png",
        dpi=250,
        bbox_inches="tight"
    )

    plt.close(fig)


# ==========================================
# 4. GENERAR RANKING
# ==========================================

def generar_ranking(tabla):

    ranking = tabla.sort_values(
        by=["modelo", "f1_score"],
        ascending=[True, False]
    )

    ranking.to_csv(
        SALIDA / "ranking_optimizadores.csv",
        index=False,
        encoding="utf-8-sig"
    )

    for modelo in MODELOS:

        print(
            f"\n=== {modelo.upper()} ==="
        )

        subset = ranking[
            ranking["modelo"] == modelo
        ]

        columnas = [
            "optimizador",
            "perdida",
            "exactitud",
            "precision",
            "sensibilidad",
            "f1_score"
        ]

        print(
            subset[columnas]
            .round(4)
            .to_string(index=False)
        )

        mejor = subset.iloc[0]

        print(
            f"\nMejor optimizador por F1: "
            f"{mejor['optimizador']}"
        )


# ==========================================
# 5. FUNCIÓN PRINCIPAL
# ==========================================

def main():

    SALIDA.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n=== CARGANDO RESULTADOS ===")

    tabla = cargar_resultados()

    print(
        f"Combinaciones cargadas: {len(tabla)}"
    )

    # Guardar tabla completa
    tabla.to_csv(
        SALIDA / "comparacion_completa.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Generar ranking
    generar_ranking(tabla)

    # Graficar las cuatro métricas
    for metrica in METRICAS:
        graficar_comparacion(
            tabla,
            metrica
        )

    # Comparar pérdidas
    graficar_comparacion(
        tabla,
        "perdida"
    )

    print(
        f"\nResultados guardados en:\n{SALIDA}"
    )


if __name__ == "__main__":
    main()
