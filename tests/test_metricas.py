from src.comun.metricas import calcular_metricas


def test_metricas_prediccion_perfecta():
    metricas = calcular_metricas([0, 1, 0, 1], [0, 1, 0, 1])
    assert all(valor == 1.0 for valor in metricas.values())
