from sklearn.linear_model import LogisticRegression
from src.config import SEMILLA


class ModeloRegresionLogistica:
    @staticmethod
    def crear(**parametros) -> LogisticRegression:
        config = {"max_iter": 2000, "class_weight": "balanced", "random_state": SEMILLA}
        config.update(parametros)
        return LogisticRegression(**config)
