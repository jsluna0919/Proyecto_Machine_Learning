
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score
)

from src.punto_2.no_supervisado.kmeans import (
    AnalisisKMeansMagic
)


class AnalisisDBSCANMagic:

    def __init__(self, semilla=42):

        self.semilla = semilla

        self.salida = (
            Path(__file__).resolve().parents[3]
            / "resultados"
            / "punto_2"
            / "no_supervisado"
            / "dbscan"
        )

    def preparar_datos(self):

        # Exactamente la misma muestra de K-Means
        preparador = AnalisisKMeansMagic(
            semilla=self.semilla,
            tamano_muestra=3000
        )

        return preparador.preparar_datos()

    def evaluar(self, X, etiquetas):

        mascara = etiquetas != -1

        X_validos = X[mascara]
        grupos = etiquetas[mascara]

        clusters = np.unique(grupos)
        cantidad = len(clusters)

        cobertura = float(np.mean(mascara))

        resultado = {
            "clusters": cantidad,
            "cobertura": cobertura,
            "ruido_pct": (1 - cobertura) * 100,
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan
        }

        if not (
            2 <= cantidad < len(X_validos)
        ):
            return resultado

        # Las métricas internas se calculan
        # solamente para los puntos agrupados.
        try:

            resultado["silhouette"] = silhouette_score(
                X_validos,
                grupos,
                sample_size=min(
                    1000,
                    len(X_validos)
                ),
                random_state=self.semilla
            )

        except ValueError:
            pass

        resultado["calinski_harabasz"] = (
            calinski_harabasz_score(
                X_validos,
                grupos
            )
        )

        resultado["davies_bouldin"] = (
            davies_bouldin_score(
                X_validos,
                grupos
            )
        )

        return resultado

    def graficar(self, X, etiquetas):

        # PCA solamente para visualización
        pca = PCA(n_components=2)

        X_visual = pca.fit_transform(X)

        mascara = etiquetas != -1

        plt.figure(figsize=(9, 6))

        # Puntos que pertenecen a clusters
        plt.scatter(
            X_visual[mascara, 0],
            X_visual[mascara, 1],
            c=etiquetas[mascara],
            cmap="tab10",
            s=12,
            alpha=0.65,
            label="Clusters"
        )

        # Ruido identificado por DBSCAN
        plt.scatter(
            X_visual[~mascara, 0],
            X_visual[~mascara, 1],
            c="lightgray",
            s=12,
            alpha=0.5,
            label="Ruido (-1)"
        )

        plt.title("DBSCAN - Agrupamiento MAGIC")
        plt.xlabel("Componente principal 1")
        plt.ylabel("Componente principal 2")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            self.salida / "clusters_dbscan.png",
            dpi=250
        )

        plt.close()

    def ejecutar(self):

        self.salida.mkdir(
            parents=True,
            exist_ok=True
        )

        X, y_real = self.preparar_datos()

        print("\n=== DBSCAN MAGIC ===")
        print(f"Dimensiones: {X.shape}")

        resultados = []
        configuraciones = {}

        # Explorar escalas de vecindad
        min_samples_opciones = [5, 10, 20]

        cuantiles = [
            0.25,
            0.40,
            0.55,
            0.70,
            0.85,
            0.95
        ]

        for min_samples in min_samples_opciones:

            vecinos = NearestNeighbors(
                n_neighbors=min_samples
            )

            vecinos.fit(X)

            distancias, _ = vecinos.kneighbors(X)

            distancias_k = distancias[:, -1]

            for cuantil in cuantiles:

                eps = float(
                    np.quantile(
                        distancias_k,
                        cuantil
                    )
                )

                modelo = DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    n_jobs=-1
                )

                # Entrenar únicamente con X
                etiquetas = modelo.fit_predict(X)

                metricas = self.evaluar(
                    X,
                    etiquetas
                )

                registro = {
                    "min_samples": min_samples,
                    "cuantil": cuantil,
                    "eps": eps,
                    **metricas
                }

                resultados.append(registro)

                clave = (
                    min_samples,
                    cuantil
                )

                configuraciones[clave] = etiquetas

                print(
                    f"min_samples={min_samples:2d} | "
                    f"eps={eps:.4f} | "
                    f"clusters={metricas['clusters']:3d} | "
                    f"ruido={metricas['ruido_pct']:6.2f}% | "
                    f"Silhouette={metricas['silhouette']:.4f}",
                    flush=True
                )

        # Guardar todas las combinaciones
        tabla = pd.DataFrame(resultados)

        tabla.to_csv(
            self.salida / "metricas_dbscan.csv",
            index=False,
            encoding="utf-8-sig"
        )

        # Evitar escoger una configuración con
        # excelente Silhouette, pero que descarte
        # casi todos los registros.
        candidatas = tabla[
            (tabla["clusters"] >= 2)
            & (tabla["cobertura"] >= 0.70)
            & (tabla["silhouette"].notna())
        ]

        print("\n=== SELECCIÓN DE CONFIGURACIÓN ===")

        if candidatas.empty:

            print(
                "No se encontró una configuración "
                "con al menos 2 clusters y "
                "70% de cobertura."
            )

            print(
                "\nSe conservaron todas las métricas "
                "para analizar este resultado."
            )

            print(
                f"\nResultados guardados en:\n"
                f"{self.salida}"
            )

            return

        mejor = candidatas.sort_values(
            "silhouette",
            ascending=False
        ).iloc[0]

        clave = (
            int(mejor["min_samples"]),
            float(mejor["cuantil"])
        )

        etiquetas_finales = configuraciones[
            clave
        ]

        print(
            f"min_samples: {int(mejor['min_samples'])}"
        )
        print(f"eps: {mejor['eps']:.4f}")
        print(f"Clusters: {int(mejor['clusters'])}")
        print(f"Cobertura: {mejor['cobertura']:.4f}")
        print(
            f"Ruido: {mejor['ruido_pct']:.2f}%"
        )
        print(
            f"Silhouette: {mejor['silhouette']:.4f}"
        )

        # Comparación externa únicamente
        # después de seleccionar parámetros
        ari = adjusted_rand_score(
            y_real,
            etiquetas_finales
        )

        print(
            f"ARI externo (incluyendo ruido): "
            f"{ari:.4f}"
        )

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

        pd.DataFrame([{
            **mejor.to_dict(),
            "ari_externo": ari
        }]).to_csv(
            self.salida / "mejor_configuracion.csv",
            index=False
        )

        self.graficar(
            X,
            etiquetas_finales
        )

        print(
            f"\nResultados guardados en:\n"
            f"{self.salida}"
        )


if __name__ == "__main__":

    analisis = AnalisisDBSCANMagic()
    analisis.ejecutar()
