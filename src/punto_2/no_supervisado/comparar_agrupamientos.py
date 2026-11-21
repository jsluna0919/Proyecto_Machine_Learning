
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

RAIZ = Path(__file__).resolve().parents[3]

BASE = (
    RAIZ
    / "resultados"
    / "punto_2"
    / "no_supervisado"
)

SALIDA = BASE / "comparacion"

METODOS = {
    "K-Means": "kmeans",
    "Jerarquico Ward": "jerarquico",
    "DBSCAN": "dbscan"
}


# ==========================================
# 2. CARGAR CONFIGURACIÓN SELECCIONADA
# ==========================================

def obtener_configuracion(nombre, carpeta):

    ruta = BASE / carpeta

    if nombre == "DBSCAN":

        tabla = pd.read_csv(
            ruta / "mejor_configuracion.csv"
        )

        return tabla.iloc[0]

    archivo = (
        "metricas_kmeans.csv"
        if nombre == "K-Means"
        else "metricas_jerarquico.csv"
    )

    tabla = pd.read_csv(ruta / archivo)

    # Misma selección usada anteriormente
    return tabla.loc[
        tabla["silhouette"].idxmax()
    ]


# ==========================================
# 3. ANALIZAR TAMAÑOS DE LOS CLUSTERS
# ==========================================

def analizar_composicion(carpeta):

    ruta = (
        BASE
        / carpeta
        / "composicion_clusters.csv"
    )

    tabla = pd.read_csv(ruta)

    tabla["Cluster"] = tabla["Cluster"].astype(int)

    tabla["Total"] = (
        tabla["Gamma"]
        + tabla["Hadron"]
    )

    total = int(tabla["Total"].sum())

    # Los registros -1 representan ruido
    grupos = tabla[
        tabla["Cluster"] != -1
    ]

    ruido = tabla[
        tabla["Cluster"] == -1
    ]

    n_agrupados = int(grupos["Total"].sum())
    n_ruido = int(ruido["Total"].sum())

    menor = int(grupos["Total"].min())
    mayor = int(grupos["Total"].max())

    porcentaje_menor = (
        100 * menor / n_agrupados
    )

    estadisticas = {
        "clusters": len(grupos),
        "total": total,
        "agrupados": n_agrupados,
        "ruido": n_ruido,
        "cobertura_pct": (
            100 * n_agrupados / total
        ),
        "ruido_pct": (
            100 * n_ruido / total
        ),
        "cluster_minimo": menor,
        "cluster_maximo": mayor,
        "cluster_minimo_pct": porcentaje_menor
    }

    return tabla, estadisticas


# ==========================================
# 4. CONSOLIDAR RESULTADOS
# ==========================================

def comparar():

    resultados = []
    composiciones = []

    for nombre, carpeta in METODOS.items():

        configuracion = obtener_configuracion(
            nombre,
            carpeta
        )

        composicion, estadisticas = (
            analizar_composicion(carpeta)
        )

        if estadisticas["total"] != 3000:
            raise ValueError(
                f"{nombre}: se esperaban 3000 "
                "observaciones."
            )

        composicion["Metodo"] = nombre
        composiciones.append(composicion)

        resultado = {
            "metodo": nombre,
            "clusters": estadisticas["clusters"],
            "silhouette": configuracion["silhouette"],
            "calinski_harabasz": configuracion[
                "calinski_harabasz"
            ],
            "davies_bouldin": configuracion[
                "davies_bouldin"
            ],
            "cobertura_pct": estadisticas[
                "cobertura_pct"
            ],
            "ruido_pct": estadisticas["ruido_pct"],
            "cluster_minimo": estadisticas[
                "cluster_minimo"
            ],
            "cluster_maximo": estadisticas[
                "cluster_maximo"
            ],
            "cluster_minimo_pct": estadisticas[
                "cluster_minimo_pct"
            ],
            "ari_externo": configuracion[
                "ari_externo"
            ]
        }

        if nombre == "DBSCAN":
            resultado["configuracion"] = (
                f"eps={configuracion['eps']:.4f}, "
                f"min_samples="
                f"{int(configuracion['min_samples'])}"
            )
        else:
            resultado["configuracion"] = (
                f"K={int(configuracion['k'])}"
            )

        # Advertir sobre grupos marginales
        resultado["alerta_cluster_pequeno"] = (
            estadisticas["cluster_minimo_pct"] < 5
        )

        resultados.append(resultado)

    resumen = pd.DataFrame(resultados)

    detalles = pd.concat(
        composiciones,
        ignore_index=True
    )

    return resumen, detalles


# ==========================================
# 5. CREAR GRÁFICAS
# ==========================================

def graficar(resumen):

    # Gráfica de Silhouette registrado
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        resumen["metodo"],
        resumen["silhouette"]
    )

    ax.set_ylabel("Silhouette")
    ax.set_title(
        "Silhouette registrado por método"
    )

    ax.text(
        0.5,
        -0.18,
        "Nota: DBSCAN excluye el ruido; "
        "las métricas no son totalmente equivalentes.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9
    )

    fig.tight_layout()

    fig.savefig(
        SALIDA / "comparacion_silhouette.png",
        dpi=250,
        bbox_inches="tight"
    )

    plt.close(fig)

    # Gráfica del tamaño del cluster más pequeño
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        resumen["metodo"],
        resumen["cluster_minimo_pct"]
    )

    ax.set_ylabel(
        "Porcentaje de los registros agrupados (%)"
    )

    ax.set_title(
        "Representatividad del cluster más pequeño"
    )

    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()

    fig.savefig(
        SALIDA / "balance_clusters.png",
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

    resumen, detalles = comparar()

    resumen.to_csv(
        SALIDA / "resumen_agrupamientos.csv",
        index=False,
        encoding="utf-8-sig"
    )

    detalles.to_csv(
        SALIDA / "composicion_agrupamientos.csv",
        index=False,
        encoding="utf-8-sig"
    )

    graficar(resumen)

    print("\n=== COMPARACIÓN NO SUPERVISADA ===")

    columnas = [
        "metodo",
        "clusters",
        "silhouette",
        "ari_externo",
        "cobertura_pct",
        "cluster_minimo",
        "cluster_minimo_pct"
    ]

    print(
        resumen[columnas]
        .round(4)
        .to_string(index=False)
    )

    print("\n=== ADVERTENCIAS ===")

    for _, fila in resumen.iterrows():

        if fila["alerta_cluster_pequeno"]:

            print(
                f"{fila['metodo']}: "
                "se detectó un cluster marginal "
                "con menos del 5% de los "
                "registros agrupados."
            )

    print(
        "\nLa comparación de Silhouette "
        "requiere considerar el ruido y "
        "la distribución de tamaños."
    )

    print(
        f"\nResultados guardados en:\n{SALIDA}"
    )


if __name__ == "__main__":
    main()
