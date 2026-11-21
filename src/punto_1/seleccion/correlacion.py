import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class SelectorCorrelacion(BaseEstimator, TransformerMixin):
    """Retiene las k variables de mayor correlación absoluta con y.

    Solo se debe ajustar con entrenamiento. Se usa dentro de Pipeline.
    """

    def __init__(self, k: int = 20):
        self.k = k

    def fit(self, X, y):
        if y is None:
            raise ValueError("El filtro de correlación requiere y.")
        a = np.asarray(X, dtype=float)
        y_array = np.asarray(y, dtype=float).reshape(-1)
        if a.ndim != 2 or len(y_array) != len(a):
            raise ValueError("Dimensiones incompatibles entre X e y")
        if not 1 <= self.k <= a.shape[1]:
            raise ValueError("k debe estar entre 1 y el número de variables")
        if not np.isfinite(a).all() or not np.isfinite(y_array).all():
            raise ValueError("Hay valores faltantes o infinitos; depura primero")
        xc = a - a.mean(axis=0)
        yc = y_array - y_array.mean()
        denom = np.sqrt((xc**2).sum(axis=0) * (yc**2).sum())
        correlaciones = np.divide(
            (xc * yc[:, None]).sum(axis=0),
            denom,
            out=np.zeros(a.shape[1], dtype=float),
            where=denom != 0,
        )
        self.importancias_ = np.abs(correlaciones)
        self.indices_ = np.sort(np.argsort(-self.importancias_)[: self.k])
        self.n_features_in_ = a.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "indices_")
        a = np.asarray(X, dtype=float)
        if a.ndim != 2 or a.shape[1] != self.n_features_in_:
            raise ValueError("El número de características cambió")
        return a[:, self.indices_]
