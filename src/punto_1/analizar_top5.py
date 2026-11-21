"""Analiza los 32 experimentos ya validados, sin volver a entrenarlos.

Uso:
    python -m src.punto_1.analizar_top5
    python -m src.punto_1.analizar_top5 --cv 3

Genera CSV y PNG en resultados/punto_1/. NO utiliza el conjunto de prueba.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import RESULTADOS_PUNTO_1


def cargar_ranking(cv_folds: int) -> pd.DataFrame:
    ruta = RESULTADOS_PUNTO_1 / "validacion_cruzada.csv"
    if not ruta.is_file():
        raise FileNotFoundError(f"No encuentro los resultados: {ruta}")

    tabla = pd.read_csv(ruta)
    columnas_necesarias = {
        "modelo_id", "modelo", "seleccion", "cv_folds", "f1_macro",
        "f1_macro_std", "precision_macro", "sensibilidad_fraude",
        "exactitud", "verdaderos_negativos", "falsos_positivos",
        "falsos_negativos", "verdaderos_positivos",
    }
    faltantes = columnas_necesarias - set(tabla.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en el CSV: {sorted(faltantes)}")

    tabla = tabla.loc[tabla["cv_folds"] == cv_folds].copy()
    if tabla.empty:
        raise ValueError(f"No existen experimentos con {cv_folds} folds")

    tabla = tabla.sort_values(
        ["f1_macro", "sensibilidad_fraude"], ascending=[False, False]
    ).drop_duplicates(subset=["modelo_id"], keep="first")
    return tabla.reset_index(drop=True)


def graficar_top5(top5: pd.DataFrame, carpeta):
    nombres = top5["modelo_id"].str.replace("_", " ", regex=False)
    x = np.arange(len(top5))
    ancho = 0.36

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - ancho / 2, top5["f1_macro"], ancho, label="F1 macro")
    ax.bar(x + ancho / 2, top5["sensibilidad_fraude"], ancho,
           label="Sensibilidad fraude")
    ax.set_xticks(x, nombres, rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Valor de la métrica")
    ax.set_title("Top 5 - validación cruzada")
    ax.legend()
    fig.tight_layout()
    fig.savefig(carpeta / "top5_metricas_cv.png", dpi=200)
    plt.close(fig)


def graficar_familias(mejores: pd.DataFrame, carpeta):
    x = np.arange(len(mejores))
    ancho = 0.36
    nombres = mejores["modelo"].str.replace("_", " ", regex=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - ancho / 2, mejores["f1_macro"], ancho, label="F1 macro")
    ax.bar(x + ancho / 2, mejores["sensibilidad_fraude"], ancho,
           label="Sensibilidad fraude")
    ax.set_xticks(x, nombres)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Valor de la métrica")
    ax.set_title("Mejor configuración de cada algoritmo (validación cruzada)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(carpeta / "comparacion_algoritmos_cv.png", dpi=200)
    plt.close(fig)


def graficar_matriz_cv(mejor: pd.Series, carpeta):
    # Conteos acumulados de predicciones fuera de muestra en los folds.
    matriz = np.array([
        [int(mejor["verdaderos_negativos"]), int(mejor["falsos_positivos"])],
        [int(mejor["falsos_negativos"]), int(mejor["verdaderos_positivos"])],
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    imagen = ax.imshow(matriz)
    fig.colorbar(imagen, ax=ax)
    ax.set_xticks([0, 1], ["Normal", "Fraude"])
    ax.set_yticks([0, 1], ["Normal", "Fraude"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase real")
    ax.set_title(f"Matriz CV acumulada: {mejor['modelo_id']}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matriz[i, j]:,}", ha="center", va="center",
                    color="white" if matriz[i, j] > matriz.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(carpeta / "matriz_cv_mejor_modelo.png", dpi=200)
    plt.close(fig)
    return matriz


def main():
    parser = argparse.ArgumentParser(description="Analiza top 5 de validación cruzada")
    parser.add_argument("--cv", type=int, default=3, help="Número de folds utilizado")
    args = parser.parse_args()

    tabla = cargar_ranking(args.cv)
    top5 = tabla.head(5).copy()
    mejores = tabla.drop_duplicates(subset=["modelo"], keep="first").copy()

    carpeta = RESULTADOS_PUNTO_1
    carpeta.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(carpeta / "ranking_completo_cv.csv", index=False, encoding="utf-8-sig")
    top5.to_csv(carpeta / "top5_cv.csv", index=False, encoding="utf-8-sig")
    mejores.to_csv(carpeta / "mejores_por_algoritmo_cv.csv", index=False,
                   encoding="utf-8-sig")

    graficar_top5(top5, carpeta)
    graficar_familias(mejores, carpeta)
    matriz = graficar_matriz_cv(top5.iloc[0], carpeta)

    print(f"\nModelos evaluados con {args.cv} folds: {len(tabla)}")
    if len(tabla) < 30:
        print("AVISO: hay menos de 30 configuraciones registradas.")
    columnas = ["modelo_id", "f1_macro", "f1_macro_std", "sensibilidad_fraude"]
    print("\n=== TOP 5 (VALIDACIÓN CRUZADA) ===")
    print(top5[columnas].round(4).to_string(index=False))
    print("\n=== MEJOR MODELO DE CADA FAMILIA ===")
    print(mejores[columnas].round(4).to_string(index=False))
    print("\nMatriz de confusión CV acumulada del mejor modelo:")
    print(matriz)
    print(f"\nArchivos CSV y PNG guardados en: {carpeta.resolve()}")
    print("Nota: todavía NO se ha utilizado el conjunto de prueba reservado.")


if __name__ == "__main__":
    main()