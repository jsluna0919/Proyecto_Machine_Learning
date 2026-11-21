from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DATOS_ORIGINALES = RAIZ / "data" / "raw"
DATOS_PROCESADOS = RAIZ / "data" / "processed"
RESULTADOS_PUNTO_1 = RAIZ / "resultados" / "punto_1"
RESULTADOS_PUNTO_2 = RAIZ / "resultados" / "punto_2"
CREDITCARD = DATOS_ORIGINALES / "creditcard.csv"
MAGIC = DATOS_ORIGINALES / "magic.csv"
SEMILLA = 42
