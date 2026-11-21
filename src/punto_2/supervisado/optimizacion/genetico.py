
import numpy as np


class AlgoritmoGenetico:
    """
    Optimizador mediante algoritmo genético.

    Utiliza:
    - Selección por torneo
    - Cruce aritmético
    - Mutación gaussiana
    - Elitismo
    """

    def __init__(
        self,
        poblacion=20,
        generaciones=40,
        prob_cruce=0.85,
        prob_mutacion=0.15,
        sigma_mutacion=0.15,
        escala_inicial=0.5,
        elitismo=2,
        semilla=42
    ):
        self.poblacion = poblacion
        self.generaciones = generaciones
        self.prob_cruce = prob_cruce
        self.prob_mutacion = prob_mutacion
        self.sigma_mutacion = sigma_mutacion
        self.escala_inicial = escala_inicial
        self.elitismo = elitismo
        self.semilla = semilla

        self.historial = []
        self.evaluaciones = 0
        self.mejores_parametros = None

    def optimizar(self, modelo, X, y):

        if self.poblacion < 4:
            raise ValueError(
                "La población debe ser al menos 4"
            )

        if not 1 <= self.elitismo < self.poblacion:
            raise ValueError(
                "El elitismo debe estar entre 1 "
                "y población - 1"
            )

        if self.generaciones < 1:
            raise ValueError(
                "Debe existir al menos una generación"
            )

        rng = np.random.default_rng(self.semilla)

        # Parámetros iniciales del modelo
        centro = modelo.parametros_iniciales(
            self.semilla
        )

        # Crear población de soluciones
        poblacion = (
            centro
            + rng.normal(
                0,
                self.escala_inicial,
                size=(self.poblacion, centro.size)
            )
        )

        poblacion[0] = centro.copy()

        mejor_perdida = np.inf
        mejor_individuo = None

        self.historial = []
        self.evaluaciones = 0

        def seleccionar_padre(puntuaciones):
            """Selección mediante torneo de 3 individuos."""

            candidatos = rng.choice(
                self.poblacion,
                size=3,
                replace=False
            )

            ganador = candidatos[
                np.argmin(puntuaciones[candidatos])
            ]

            return poblacion[ganador]

        print(
            f"\nIniciando algoritmo genético: "
            f"{modelo.__class__.__name__}",
            flush=True
        )

        # Evolución
        for generacion in range(
            1, self.generaciones + 1
        ):

            # Evaluar individuos
            puntuaciones = np.array([
                modelo.perdida(individuo, X, y)
                for individuo in poblacion
            ])

            self.evaluaciones += len(poblacion)

            if not np.all(np.isfinite(puntuaciones)):
                raise ValueError(
                    "Se encontraron pérdidas no finitas"
                )

            # Buscar mejor individuo de la generación
            indice = int(np.argmin(puntuaciones))

            if puntuaciones[indice] < mejor_perdida:
                mejor_perdida = float(
                    puntuaciones[indice]
                )

                mejor_individuo = (
                    poblacion[indice].copy()
                )

            # Registrar mejor pérdida global
            self.historial.append(mejor_perdida)

            if (
                generacion == 1
                or generacion % 10 == 0
                or generacion == self.generaciones
            ):
                print(
                    f"Generación {generacion:3d}/"
                    f"{self.generaciones} | "
                    f"Mejor pérdida: {mejor_perdida:.6f}",
                    flush=True
                )

            if generacion == self.generaciones:
                break

            # Conservar mejores individuos (elitismo)
            orden = np.argsort(puntuaciones)

            nueva_poblacion = [
                poblacion[i].copy()
                for i in orden[:self.elitismo]
            ]

            # Crear descendientes
            while len(nueva_poblacion) < self.poblacion:

                padre1 = seleccionar_padre(
                    puntuaciones
                )

                padre2 = seleccionar_padre(
                    puntuaciones
                )

                # Cruce
                if rng.random() < self.prob_cruce:

                    alpha = rng.random(
                        centro.size
                    )

                    hijo = (
                        alpha * padre1
                        + (1 - alpha) * padre2
                    )
                else:
                    hijo = padre1.copy()

                # Mutación
                mascara = (
                    rng.random(centro.size)
                    < self.prob_mutacion
                )

                hijo[mascara] += rng.normal(
                    0,
                    self.sigma_mutacion,
                    size=int(mascara.sum())
                )

                nueva_poblacion.append(hijo)

            poblacion = np.array(
                nueva_poblacion
            )

        self.mejores_parametros = (
            mejor_individuo.copy()
        )

        print(
            f"Optimización finalizada | "
            f"Mejor pérdida: {mejor_perdida:.6f}"
        )

        print(
            f"Evaluaciones de la función objetivo: "
            f"{self.evaluaciones}"
        )

        return self.mejores_parametros
