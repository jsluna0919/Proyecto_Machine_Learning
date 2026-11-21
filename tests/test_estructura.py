from src.punto_1.experimentos import generar_experimentos, construir_pipeline


def test_hay_32_experimentos():
    experimentos = generar_experimentos()
    assert len(experimentos) == 32
    assert len({e.identificador for e in experimentos}) == 32
    assert {e.modelo for e in experimentos} == {"logistica", "red_neuronal", "svm", "arbol"}
    assert {e.seleccion for e in experimentos} == {"correlacion", "pca"}


def test_pipeline():
    exp = generar_experimentos()[0]
    pipeline = construir_pipeline(exp)
    assert list(pipeline.named_steps) == ["escalado", "seleccion", "modelo"]
