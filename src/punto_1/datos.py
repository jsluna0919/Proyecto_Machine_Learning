import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CREDITCARD, SEMILLA


def cargar_datos(ruta=CREDITCARD):
    """Lee CSV y separa características X y etiqueta y, sin modificar datos."""
    if not ruta.is_file():
        raise FileNotFoundError(f"No encuentro el CSV: {ruta}")
    df = pd.read_csv(ruta)
    if "Class" not in df.columns:
        raise ValueError("Falta la columna objetivo 'Class'")
    X = df.drop(columns=["Class"])
    y = df["Class"]
    if set(y.dropna().unique()) != {0, 1}:
        raise ValueError("Se esperaban ambas clases 0 y 1 en 'Class'.")
    return X, y


def dividir_datos(X, y, test_size=0.2):
    return train_test_split(
        X, y, test_size=test_size, random_state=SEMILLA, stratify=y
    )
