from sklearn.tree import DecisionTreeClassifier
from src.config import SEMILLA


class ModeloArbolDecision:
    @staticmethod
    def crear(**parametros) -> DecisionTreeClassifier:
        config = {"max_depth": 5, "class_weight": "balanced", "random_state": SEMILLA}
        config.update(parametros)
        return DecisionTreeClassifier(**config)
