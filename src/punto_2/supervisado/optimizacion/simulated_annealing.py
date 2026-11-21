
import numpy as np


class SimulatedAnnealing:
    """
    Optimizador mediante recocido simulado.

    Compatible con modelos que implementen:
        parametros_iniciales()
        perdida()

    Utiliza enfriamiento geométrico y
    perturbaciones gaussianas.
    """

    def __init__(
        self,
        evaluaciones=800,
        temperatura_inicial=1.0,
        temperatura_final=0.001,
        sigma_mutacion=0.3,
        semilla=42
    ):
        if evaluaciones < 2:
            raise ValueError(
                "Se necesitan al menos 2 evaluaciones."
            )

        if (
            temperatura_inicial <= 0
            or temperatura_final <= 0
            or temperatura_final > temperatura_inicial
        ):
            raise ValueError(
                "Las temperaturas deben ser positivas "
                "y la final no puede superar la inicial."
            )

        if sigma_mutacion <= 0:
            raise ValueError(
                "sigma_mutacion debe ser positivo."
            )

        self.max_evaluaciones = evaluaciones
        self.temperatura_inicial = temperatura_inicial
        self.temperatura_final = temperatura_final
        self.sigma_mutacion = sigma_mutacion
        self.semilla = semilla

        self.historial = []
        self.historial_actual = []
        self.evaluaciones = 0
        self.aceptados = 0
        self.mejores_parametros = None

    def optimizar(self, modelo, X, y):

        rng = np.random.default_rng(self.semilla)

        # ----------------------------------
        # 1. Inicialización
        # ----------------------------------

        actual = modelo.parametros_iniciales(
            semilla=self.semilla
        ).copy()

        perdida_actual = modelo.perdida(
            actual, X, y
        )

        if not np.isfinite(perdida_actual):
            raise ValueError(
                "La pérdida inicial no es finita."
            )

        mejor = actual.copy()
        mejor_perdida = float(perdida_actual)

        self.evaluaciones = 1
        self.aceptados = 0

        self.historial = [mejor_perdida]
        self.historial_actual = [float(perdida_actual)]

        print(
            f"\nIniciando Simulated Annealing: "
            f"{modelo.__class__.__name__}",
            flush=True
        )

        print(
            f"Evaluación 1/{self.max_evaluaciones} | "
            f"Mejor pérdida: {mejor_perdida:.6f}",
            flush=True
        )

        # ----------------------------------
        # 2. Proceso de optimización
        # ----------------------------------

        for paso in range(
            1, self.max_evaluaciones
        ):

            # Enfriamiento geométrico
            fraccion = paso / (
                self.max_evaluaciones - 1
            )

            temperatura = (
                self.temperatura_inicial
                * (
                    self.temperatura_final
                    / self.temperatura_inicial
                ) ** fraccion
            )

            # Reducir gradualmente la escala
            # de las perturbaciones
            factor = (
                0.15
                + 0.85
                * temperatura
                / self.temperatura_inicial
            )

            sigma = self.sigma_mutacion * factor

            # Proponer nueva solución
            candidato = actual + rng.normal(
                0,
                sigma,
                size=actual.shape
            )

            # Evaluar candidato
            perdida_candidato = modelo.perdida(
                candidato, X, y
            )

            self.evaluaciones += 1

            if not np.isfinite(perdida_candidato):
                raise ValueError(
                    "Se obtuvo una pérdida no finita."
                )

            # Diferencia respecto a la solución actual
            diferencia = (
                perdida_candidato - perdida_actual
            )

            # ----------------------------------
            # 3. Criterio de aceptación
            # ----------------------------------

            aceptar = False

            if diferencia <= 0:
                # Siempre aceptar mejoras
                aceptar = True

            else:
                # Aceptar empeoramientos con
                # probabilidad dependiente de T
                probabilidad = np.exp(
                    -diferencia / temperatura
                )

                if rng.random() < probabilidad:
                    aceptar = True

            if aceptar:
                actual = candidato.copy()
                perdida_actual = float(
                    perdida_candidato
                )
                self.aceptados += 1

            # ----------------------------------
            # 4. Mejor solución global
            # ----------------------------------

            if perdida_actual < mejor_perdida:
                mejor_perdida = perdida_actual
                mejor = actual.copy()

            self.historial.append(mejor_perdida)

            self.historial_actual.append(
                perdida_actual
            )

            if (
                self.evaluaciones % 100 == 0
                or self.evaluaciones
                == self.max_evaluaciones
            ):
                print(
                    f"Evaluación "
                    f"{self.evaluaciones}/"
                    f"{self.max_evaluaciones} | "
                    f"Temperatura: "
                    f"{temperatura:.5f} | "
                    f"Mejor pérdida: "
                    f"{mejor_perdida:.6f}",
                    flush=True
                )

        # ----------------------------------
        # 5. Resultados
        # ----------------------------------

        self.mejores_parametros = mejor.copy()

        print(
            f"\nOptimización finalizada | "
            f"Mejor pérdida: {mejor_perdida:.6f}",
            flush=True
        )

        print(
            f"Evaluaciones de la función objetivo: "
            f"{self.evaluaciones}"
        )

        print(
            f"Movimientos aceptados: "
            f"{self.aceptados}"
        )

        return self.mejores_parametros
