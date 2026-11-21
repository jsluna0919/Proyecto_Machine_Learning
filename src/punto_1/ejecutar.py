"""Ejecutor inicial de demostración; la metodología final aún debe completarse."""
import argparse
import pandas as pd

from src.config import RESULTADOS_PUNTO_1
from src.punto_1.datos import cargar_datos, dividir_datos
from src.punto_1.entrenamiento import entrenar_y_evaluar
from src.punto_1.experimentos import generar_experimentos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-modelos", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.max_modelos <= 32:
        parser.error("--max-modelos debe estar entre 1 y 32")

    X, y = cargar_datos()
    print(f"Datos cargados: {len(X)} registros; {X.shape[1]} características")
    print(f"Distribución de clases:\n{y.value_counts().to_string()}")
    if X.isna().any().any():
        raise ValueError("Hay nulos en X. Examínalos antes de entrenar.")

    X_train, X_test, y_train, y_test = dividir_datos(X, y)
    resultados = []
    for experimento in generar_experimentos()[: args.max_modelos]:
        print(f"Entrenando: {experimento.identificador}")
        _, metricas = entrenar_y_evaluar(
            experimento, X_train, X_test, y_train, y_test
        )
        resultados.append({
            "modelo_id": experimento.identificador,
            "modelo": experimento.modelo,
            "seleccion": experimento.seleccion,
            **metricas,
        })
    RESULTADOS_PUNTO_1.mkdir(parents=True, exist_ok=True)
    destino = RESULTADOS_PUNTO_1 / "metricas_iniciales.csv"
    pd.DataFrame(resultados).to_csv(destino, index=False)
    print(f"Resultados guardados en: {destino}")
    print("Nota: falta CV, selección robusta de parámetros y análisis del top 5.")


if __name__ == "__main__":
    main()
