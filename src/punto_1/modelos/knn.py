from sklearn.neighbors import KNeighborsClassifier


class ModeloKNN:
    """Modelo alternativo no lineal (no entra en las 32 pruebas iniciales)."""

    @staticmethod
    def crear(**parametros) -> KNeighborsClassifier:
        config = {"n_neighbors": 5}
        config.update(parametros)
        return KNeighborsClassifier(**config)
