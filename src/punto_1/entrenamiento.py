from src.comun.metricas import calcular_metricas
from src.punto_1.experimentos import construir_pipeline, Experimento


def entrenar_y_evaluar(experimento: Experimento, X_train, X_test, y_train, y_test):
    """Ajusta un solo modelo y evalúa una vez sobre prueba (demo inicial)."""
    pipeline = construir_pipeline(experimento)
    pipeline.fit(X_train, y_train)
    predicciones = pipeline.predict(X_test)
    metricas = calcular_metricas(y_test, predicciones)
    return pipeline, metricas
