
import numpy as np
from scipy.special import expit


class RegresionLogisticaMagic:
    """
    Modelo de clasificación binaria mediante
    regresión logística.

    Los parámetros se almacenan en un vector
    para permitir diferentes optimizadores.
    """

    def __init__(self, n_entradas=10, regularizacion=0.0001):
        self.n_entradas = n_entradas
        self.regularizacion = regularizacion

    @property
    def numero_parametros(self):
        return self.n_entradas + 1

    def parametros_iniciales(self, semilla=42):
        rng = np.random.default_rng(semilla)

        return rng.normal(
            0,
            0.01,
            size=self.numero_parametros
        )

    def probabilidades(self, X, parametros):
        X = np.asarray(X, dtype=float)

        w = parametros[:-1]
        b = parametros[-1]

        z = X @ w + b

        return expit(z)

    def predecir(self, X, parametros, umbral=0.5):
        probabilidades = self.probabilidades(
            X, parametros
        )

        return (probabilidades >= umbral).astype(int)

    def perdida_y_gradiente(self, parametros, X, y):
        """
        Calcula la entropía cruzada binaria
        regularizada y su gradiente.
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        w = parametros[:-1]
        b = parametros[-1]

        z = X @ w + b
        p = expit(z)

        # Función de pérdida
        perdida = np.mean(
            np.logaddexp(0, z) - y * z
        )

        perdida += (
            self.regularizacion / 2
        ) * np.sum(w ** 2)

        # Gradiente
        diferencia = p - y

        grad_w = (
            (X.T @ diferencia) / len(y)
            + self.regularizacion * w
        )

        grad_b = np.mean(diferencia)

        gradiente = np.concatenate([
            grad_w,
            [grad_b]
        ])

        return float(perdida), gradiente

    def perdida(self, parametros, X, y):
        """
        Calcula la función de pérdida sin gradientes.
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        w = parametros[:-1]
        b = parametros[-1]

        z = X @ w + b

        entropia = np.mean(
            np.logaddexp(0, z) - y * z
        )

        regularizacion = (
            self.regularizacion / 2
        ) * np.sum(w ** 2)

        return float(entropia + regularizacion)
