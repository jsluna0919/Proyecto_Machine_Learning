from sklearn.neural_network import MLPClassifier
from src.config import SEMILLA


class ModeloRedNeuronal:
    @staticmethod
    def crear(**parametros) -> MLPClassifier:
        config = {"hidden_layer_sizes": (32,), "max_iter": 200, "random_state": SEMILLA}
        config.update(parametros)
        return MLPClassifier(**config)
