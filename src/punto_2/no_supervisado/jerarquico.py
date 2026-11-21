
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score
)

from src.punto_2.no_supervisado.kmeans import (
    AnalisisKMeansMagic
)


class AnalisisJerarquicoMagic:

    def __init__(self, semilla=42):

        self.semilla = semilla

        self.salida = (
            Path(__file__).resolve().parents[3]
            / "resultados"
            / "punto_2"
            / "no_supervisado"
            / "jerarquico"
        )

    def preparar_datos(self):
        """
        Reutiliza exactamente la misma muestra
        y estandarización utilizadas en K-Means.
        """

        preparador = AnalisisKMeansMagic(
            semilla=self.semilla,
            tamano_muestra=3000
        )

        return preparador.preparar_datos()

    def ejecutar(self):

        self.salida.mkdir(
            parents=True,
            exist_ok=True
        )

        X, y_real = self.preparar_datos()

        print("\n=== AGRUPAMIENTO JERÁRQUICO ===")
        print(f"Dimensiones: {X.shape}")
        print("Método de enlace: Ward")

        resultados = []
        agrupamientos = {}

        # Probar entre 2 y 6 grupos
        for k in range(2, 7):

            modelo = AgglomerativeClustering(
                n_clusters=k,
                linkage="ward"
            )

            # Entrenamiento sin usar etiquetas reales
            etiquetas = modelo.fit_predict(X)

            agrupamientos[k] = etiquetas

            # Métricas internas
            silhouette = silhouette_score(
                X,
                etiquetas,
                sample_size=min(1000, len(X)),
                random_state=self.semilla
            )

            calinski = calinski_harabasz_score(
                X,
                etiquetas
            )

            davies = davies_bouldin_score(
                X,
                etiquetas
            )

            # Métrica externa solamente para análisis
            ari = adjusted_rand_score(
                y_real,
                etiquetas
            )

            resultados.append({
                "k": k,
                "silhouette": silhouette,
                "calinski_harabasz": calinski,
                "davies_bouldin": davies,
                "ari_externo": ari
            })

            print(
                f"K={k} | "
                f"Silhouette={silhouette:.4f} | "
                f"Calinski={calinski:.2f} | "
                f"Davies={davies:.4f} | "
                f"ARI={ari:.4f}",
                flush=True
            )

        # Guardar métricas
        tabla = pd.DataFrame(resultados)

        tabla.to_csv(
            self.salida / "metricas_jerarquico.csv",
            index=False,
            encoding="utf-8-sig"
        )

        # Seleccionar K según Silhouette
        mejor_fila = tabla.loc[
            tabla["silhouette"].idxmax()
        ]

        mejor_k = int(mejor_fila["k"])
        etiquetas_finales = agrupamientos[mejor_k]

        print("\n=== MEJOR CONFIGURACIÓN ===")
        print(f"Mejor K: {mejor_k}")

        print(
            f"Silhouette: "
            f"{mejor_fila['silhouette']:.4f}"
        )

        # Distribución de observaciones por cluster
        valores, cantidades = np.unique(
            etiquetas_finales,
            return_counts=True
        )

        print("\n=== TAMAÑO DE CLUSTERS ===")

        for cluster, cantidad in zip(
            valores,
            cantidades
        ):
            print(
                f"Cluster {cluster}: {cantidad}"
            )

        # Comparación posterior con clases reales
        comparacion = pd.crosstab(
            pd.Series(
                etiquetas_finales,
                name="Cluster"
            ),
            pd.Series(
                np.where(
                    y_real == 1,
                    "Gamma",
                    "Hadron"
                ),
                name="Clase real"
            )
        )

        print("\n=== COMPOSICIÓN DE CLUSTERS ===")
        print(comparacion)

        comparacion.to_csv(
            self.salida / "composicion_clusters.csv"
        )

        # Visualizar en dos componentes PCA
        pca = PCA(n_components=2)

        X_visual = pca.fit_transform(X)

        plt.figure(figsize=(9, 6))

        plt.scatter(
            X_visual[:, 0],
            X_visual[:, 1],
            c=etiquetas_finales,
            cmap="tab10",
            s=12,
            alpha=0.65
        )

        plt.xlabel("Componente principal 1")
        plt.ylabel("Componente principal 2")

        plt.title(
            f"Agrupamiento jerárquico Ward - "
            f"{mejor_k} clusters"
        )

        plt.tight_layout()

        plt.savefig(
            self.salida / "clusters_jerarquico.png",
            dpi=250
        )

        plt.close()

        # Gráfica de métricas Silhouette
        plt.figure(figsize=(8, 5))

        plt.plot(
            tabla["k"],
            tabla["silhouette"],
            marker="o"
        )

        plt.xlabel("Número de clusters")
        plt.ylabel("Silhouette")

        plt.title(
            "Selección de K - Agrupamiento jerárquico"
        )

        plt.xticks(tabla["k"])
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            self.salida / "silhouette_jerarquico.png",
            dpi=250
        )

        plt.close()

        print(
            f"\nResultados guardados en:\n"
            f"{self.salida}"
        )


if __name__ == "__main__":

    analisis = AnalisisJerarquicoMagic()
    analisis.ejecutar()
