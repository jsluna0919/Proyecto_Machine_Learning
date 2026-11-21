from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from src.punto_1.datos import cargar_datos, dividir_datos
from src.punto_1.experimentos import (
    generar_experimentos,
    construir_pipeline
)
from src.comun.metricas import calcular_metricas


def main():

    # 1. Cargar y dividir los datos
    X, y = cargar_datos()
    X_train, X_test, y_train, y_test = dividir_datos(X, y)

    # 2. Crear entrenamiento y validación
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
    )

    # 3. Seleccionar dos experimentos comparables
    experimentos = [
        e for e in generar_experimentos()
        if e.modelo == "logistica"
        and e.parametros["C"] == 1.0
    ]

    # 4. Entrenar y evaluar
    for experimento in experimentos:

        print(f"\n=== {experimento.identificador} ===")

        pipeline = construir_pipeline(experimento)
        pipeline.fit(X_fit, y_fit)

        predicciones = pipeline.predict(X_val)
        metricas = calcular_metricas(y_val, predicciones)

        for nombre, valor in metricas.items():
            print(f"{nombre}: {valor:.4f}")

        print("\nMatriz de confusión:")
        print(confusion_matrix(
            y_val, predicciones, labels=[0, 1]
        ))


if __name__ == "__main__":
    main()