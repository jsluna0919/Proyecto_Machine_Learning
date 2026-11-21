
import numpy as np
from scipy.special import expit


class RegresionNeuronalMagic:
    """
    Red neuronal para clasificación binaria.

    Arquitectura:
        10 entradas
        4 neuronas ocultas (tanh)
        1 neurona de salida (sigmoide)
    """

    def __init__(
        self,
        n_entradas=10,
        n_ocultas=4,
        regularizacion=0.0001
    ):
        self.n_entradas = n_entradas
        self.n_ocultas = n_ocultas
        self.regularizacion = regularizacion

    @property
    def numero_parametros(self):
        return (
            self.n_entradas * self.n_ocultas
            + self.n_ocultas
            + self.n_ocultas
            + 1
        )

    def parametros_iniciales(self, semilla=42):
        """
        Inicializa los pesos de forma reproducible.
        """

        rng = np.random.default_rng(semilla)

        W1 = rng.normal(
            0,
            1 / np.sqrt(self.n_entradas),
            size=(self.n_entradas, self.n_ocultas)
        )

        b1 = np.zeros(self.n_ocultas)

        W2 = rng.normal(
            0,
            1 / np.sqrt(self.n_ocultas),
            size=self.n_ocultas
        )

        b2 = np.array([0.0])

        return np.concatenate([
            W1.ravel(),
            b1,
            W2,
            b2
        ])

    def _separar(self, parametros):
        """
        Convierte el vector de parámetros
        en matrices de pesos y sesgos.
        """

        parametros = np.asarray(
            parametros,
            dtype=float
        )

        if parametros.size != self.numero_parametros:
            raise ValueError(
                f"Se esperaban "
                f"{self.numero_parametros} parámetros"
            )

        fin_w1 = self.n_entradas * self.n_ocultas
        fin_b1 = fin_w1 + self.n_ocultas
        fin_w2 = fin_b1 + self.n_ocultas

        W1 = parametros[:fin_w1].reshape(
            self.n_entradas,
            self.n_ocultas
        )

        b1 = parametros[fin_w1:fin_b1]
        W2 = parametros[fin_b1:fin_w2]
        b2 = parametros[fin_w2]

        return W1, b1, W2, b2

    def _forward(self, X, parametros):
        """
        Propagación hacia adelante.
        """

        X = np.asarray(X, dtype=float)

        W1, b1, W2, b2 = self._separar(
            parametros
        )

        # Capa oculta
        activacion = np.tanh(
            X @ W1 + b1
        )

        # Capa de salida
        logits = activacion @ W2 + b2

        return activacion, logits

    def probabilidades(self, X, parametros):
        _, logits = self._forward(
            X,
            parametros
        )

        return expit(logits)

    def predecir(self, X, parametros, umbral=0.5):
        probabilidades = self.probabilidades(
            X,
            parametros
        )

        return (probabilidades >= umbral).astype(int)

    def perdida_y_gradiente(self, parametros, X, y):
        """
        Calcula entropía cruzada binaria
        y gradientes mediante backpropagation.
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        W1, b1, W2, b2 = self._separar(
            parametros
        )

        a1, z2 = self._forward(
            X,
            parametros
        )

        p = expit(z2)

        # Función de pérdida
        perdida = np.mean(
            np.logaddexp(0, z2) - y * z2
        )

        perdida += (
            self.regularizacion / 2
        ) * (
            np.sum(W1 ** 2)
            + np.sum(W2 ** 2)
        )

        # Gradiente capa de salida
        delta2 = (p - y) / len(y)

        grad_W2 = (
            a1.T @ delta2
            + self.regularizacion * W2
        )

        grad_b2 = np.sum(delta2)

        # Gradiente capa oculta
        delta1 = (
            delta2[:, None]
            * W2[None, :]
        ) * (1 - a1 ** 2)

        grad_W1 = (
            X.T @ delta1
            + self.regularizacion * W1
        )

        grad_b1 = delta1.sum(axis=0)

        # Unir gradientes en un vector
        gradiente = np.concatenate([
            grad_W1.ravel(),
            grad_b1,
            grad_W2,
            [grad_b2]
        ])

        return float(perdida), gradiente

    def perdida(self, parametros, X, y):
        """
        Calcula la pérdida neuronal sin backpropagation.
        """

        y = np.asarray(y, dtype=float).ravel()

        W1, b1, W2, b2 = self._separar(
            parametros
        )

        _, logits = self._forward(
            X, parametros
        )

        entropia = np.mean(
            np.logaddexp(0, logits) - y * logits
        )

        regularizacion = (
            self.regularizacion / 2
        ) * (
            np.sum(W1 ** 2)
            + np.sum(W2 ** 2)
        )

        return float(entropia + regularizacion)
