
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score
)

from src.punto_2.preparacion import PreparadorMagic


class AnalisisKMeansMagic:

    def __init__(self, semilla=42, tamano_muestra=3000):

        self.semilla = semilla
        self.tamano_muestra = tamano_muestra

        self.salida = (
            Path(__file__).resolve().parents[3]
            / "resultados"
            / "punto_2"
            / "no_supervisado"
            / "kmeans"
        )

    def preparar_datos(self):

        preparador = PreparadorMagic()

        X_train, _, y_train, _ = (
            preparador.dividir_datos()
        )

        # Utilizar solo datos de entrenamiento.
        # Las etiquetas no participarán
        # en el agrupamiento.
        n = min(self.tamano_muestra, len(X_train))

        X_muestra = X_train.sample(
            n=n,
            random_state=self.semilla
        )

        y_muestra = y_train.loc[
            X_muestra.index
        ].to_numpy()

        escalador = StandardScaler()

        X_escalado = escalador.fit_transform(
            X_muestra
        )

        return X_escalado, y_muestra

    def ejecutar(self):

        self.salida.mkdir(
            parents=True,
            exist_ok=True
        )

        X, y_real = self.preparar_datos()

        print("\n=== APRENDIZAJE NO SUPERVISADO ===")
        print(f"Dimensiones: {X.shape}")
        print("Algoritmo: K-Means")

        resultados = []
        agrupamientos = {}

        for k in range(2, 7):

            modelo = KMeans(
                n_clusters=k,
                random_state=self.semilla,
                n_init=10
            )

            # Aprendizaje sin utilizar y_real
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

            # Evaluación externa posterior.
            # No se utiliza para elegir k.
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
                f"ARI={ari:.4f}"
            )

        tabla = pd.DataFrame(resultados)

        tabla.to_csv(
            self.salida / "metricas_kmeans.csv",
            index=False
        )

        # Seleccionar K solo mediante Silhouette
        mejor_fila = tabla.loc[
            tabla["silhouette"].idxmax()
        ]

        mejor_k = int(mejor_fila["k"])

        etiquetas_finales = agrupamientos[
            mejor_k
        ]

        print("\n=== MEJOR CONFIGURACIÓN ===")
        print(f"Mejor K: {mejor_k}")
        print(
            f"Silhouette: "
            f"{mejor_fila['silhouette']:.4f}"
        )

        # Comparar los grupos con las clases reales
        # solo después de seleccionar K
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

        # Reducir dimensiones solo para visualizar
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

        plt.title(
            f"K-Means - {mejor_k} clusters"
        )

        plt.xlabel("Componente principal 1")
        plt.ylabel("Componente principal 2")

        plt.tight_layout()

        plt.savefig(
            self.salida / "clusters_kmeans.png",
            dpi=250
        )

        plt.close()

        print(
            f"\nResultados guardados en:\n{self.salida}"
        )


if __name__ == "__main__":
    analisis = AnalisisKMeansMagic()
    analisis.ejecutar()
