from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from src.punto_1.datos import cargar_datos, dividir_datos
from src.punto_1.experimentos import generar_experimentos, construir_pipeline
from src.comun.metricas import calcular_metricas


def obtener_resultados():
    """
    Reproduce la validación inicial y devuelve métricas
    y matrices de confusión para los dos modelos logísticos:
    correlación vs PCA.
    """

    # 1. Cargar y dividir datos
    X, y = cargar_datos()
    X_train, X_test, y_train, y_test = dividir_datos(X, y)

    # 2. Subdividir entrenamiento en fit/validación
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
    )

    # 3. Filtrar los experimentos deseados
    experimentos = [
        e for e in generar_experimentos()
        if e.modelo == "logistica" and e.parametros["C"] == 1.0
    ]

    resultados = []

    for experimento in experimentos:
        pipeline = construir_pipeline(experimento)
        pipeline.fit(X_fit, y_fit)

        predicciones = pipeline.predict(X_val)
        metricas = calcular_metricas(y_val, predicciones)
        matriz = confusion_matrix(y_val, predicciones, labels=[0, 1])

        resultados.append({
            "id": experimento.identificador,
            "metricas": metricas,
            "matriz": matriz
        })

    return resultados


def graficar_metricas(resultados, ruta_salida):
    """
    Gráfico de barras comparando métricas entre modelos.
    """

    metricas_a_mostrar = [
        "exactitud",
        "precision_macro",
        "sensibilidad_macro",
        "especificidad_macro",
        "f1_macro",
        "fowlkes_mallows_macro",
        "sensibilidad_fraude",
    ]

    etiquetas_modelos = [r["id"] for r in resultados]
    x = np.arange(len(metricas_a_mostrar))
    ancho = 0.35

    valores_1 = [resultados[0]["metricas"][m] for m in metricas_a_mostrar]
    valores_2 = [resultados[1]["metricas"][m] for m in metricas_a_mostrar]

    plt.figure(figsize=(12, 6))
    plt.bar(x - ancho / 2, valores_1, width=ancho, label=etiquetas_modelos[0])
    plt.bar(x + ancho / 2, valores_2, width=ancho, label=etiquetas_modelos[1])

    plt.xticks(x, metricas_a_mostrar, rotation=30, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Valor")
    plt.title("Comparación de métricas - Validación inicial")
    plt.legend()
    plt.tight_layout()

    archivo = ruta_salida / "comparacion_metricas_validacion_inicial.png"
    plt.savefig(archivo, dpi=300, bbox_inches="tight")
    plt.show()


def graficar_matriz_confusion(matriz, titulo, ruta_archivo):
    """
    Grafica una matriz de confusión usando matplotlib.
    """

    plt.figure(figsize=(5, 4))
    plt.imshow(matriz)
    plt.colorbar()

    etiquetas = ["Normal (0)", "Fraude (1)"]
    plt.xticks([0, 1], etiquetas)
    plt.yticks([0, 1], etiquetas)

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            plt.text(j, i, str(matriz[i, j]), ha="center", va="center")

    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(ruta_archivo, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    resultados = obtener_resultados()

    ruta_salida = Path("resultados/graficas")
    ruta_salida.mkdir(parents=True, exist_ok=True)

    # 1. Gráfico de métricas
    graficar_metricas(resultados, ruta_salida)

    # 2. Matrices de confusión
    for resultado in resultados:
        nombre_archivo = f"matriz_confusion_{resultado['id']}.png"
        ruta_archivo = ruta_salida / nombre_archivo

        graficar_matriz_confusion(
            resultado["matriz"],
            f"Matriz de confusión - {resultado['id']}",
            ruta_archivo
        )

    print("\nGráficas guardadas en:", ruta_salida.resolve())


if __name__ == "__main__":
    main()