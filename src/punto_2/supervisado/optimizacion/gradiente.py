
import numpy as np


class GradienteDescendente:
    """
    Optimizador mediante descenso por gradiente
    de lote completo (Batch Gradient Descent).

    Compatible con modelos que implementen:
        parametros_iniciales()
        perdida_y_gradiente()
    """

    def __init__(
        self,
        tasa_aprendizaje=0.1,
        iteraciones=150,
        semilla=42
    ):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.iteraciones = iteraciones
        self.semilla = semilla

        self.historial = []
        self.mejores_parametros = None

    def optimizar(self, modelo, X, y):

        # Inicializar parámetros
        parametros = modelo.parametros_iniciales(
            semilla=self.semilla
        )

        self.historial = []

        print(
            f"\nIniciando descenso por gradiente: "
            f"{modelo.__class__.__name__}",
            flush=True
        )

        for iteracion in range(1, self.iteraciones + 1):

            # Calcular pérdida y gradiente
            perdida, gradiente = (
                modelo.perdida_y_gradiente(
                    parametros,
                    X,
                    y
                )
            )

            # Comprobar estabilidad numérica
            if (
                not np.isfinite(perdida)
                or not np.all(np.isfinite(gradiente))
            ):
                raise ValueError(
                    "La optimización produjo valores no finitos."
                )

            # Actualizar los parámetros
            parametros = (
                parametros
                - self.tasa_aprendizaje * gradiente
            )

            # Registrar pérdida
            self.historial.append(perdida)

            # Mostrar avance
            if (
                iteracion == 1
                or iteracion % 25 == 0
                or iteracion == self.iteraciones
            ):
                print(
                    f"Iteración {iteracion:3d}/"
                    f"{self.iteraciones} | "
                    f"Pérdida: {perdida:.6f}",
                    flush=True
                )

        self.mejores_parametros = parametros.copy()

        # Calcular pérdida final tras la última actualización
        perdida_final, _ = modelo.perdida_y_gradiente(
            parametros,
            X,
            y
        )

        self.historial.append(perdida_final)

        print(
            f"Optimización finalizada | "
            f"Pérdida final: {perdida_final:.6f}"
        )

        return parametros
