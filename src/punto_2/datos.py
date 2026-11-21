from pathlib import Path
import pandas as pd


class DatosMagic:
    """
    Clase encargada de cargar y validar
    la base de datos MAGIC.
    """

    def __init__(self):

        self.ruta = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "raw"
            / "magic.csv"
        )

    def cargar(self):

        # Omitir la primera fila de metadatos
        df = pd.read_csv(
            self.ruta,
            skiprows=1
        )

        # Limpiar nombres de columnas
        df.columns = df.columns.str.strip()

        # Validar columna objetivo
        if "Class" not in df.columns:
            raise ValueError(
                "No existe la columna Class"
            )

        # Separar características
        X = df.drop(
            columns=["Class"]
        ).apply(pd.to_numeric, errors="raise")

        # Limpiar etiquetas
        etiquetas = df["Class"].astype("string").str.strip()

        # Validar las clases
        if etiquetas.isna().any() or not etiquetas.isin(["g", "h"]).all():
            raise ValueError(
                "Se encontraron etiquetas desconocidas"
            )

        # Codificar etiquetas
        # Gamma = 1, Hadron = 0
        y = etiquetas.map({
            "g": 1,
            "h": 0
        }).astype(int)

        # Validar nulos
        if X.isnull().sum().sum() > 0:
            raise ValueError(
                "La base de datos contiene valores nulos"
            )

        return X, y