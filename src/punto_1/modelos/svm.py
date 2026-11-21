from sklearn.svm import LinearSVC
from src.config import SEMILLA


class ModeloSVM:
    @staticmethod
    def crear(**parametros) -> LinearSVC:
        config = {"C": 1.0, "class_weight": "balanced", "random_state": SEMILLA, "max_iter": 3000}
        config.update(parametros)
        return LinearSVC(**config)
