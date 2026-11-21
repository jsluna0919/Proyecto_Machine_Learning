
import numpy as np


class OptimizadorPSO:
    """
    Optimización por enjambre de partículas.

    Compatible con modelos que implementen:
        parametros_iniciales()
        perdida()
    """

    def __init__(
        self,
        particulas=20,
        iteraciones=40,
        inercia=0.7,
        cognitivo=1.5,
        social=1.5,
        escala_inicial=0.5,
        velocidad_maxima=0.3,
        semilla=42
    ):
        if particulas < 2 or iteraciones < 1:
            raise ValueError(
                "Configuración de PSO no válida"
            )

        self.particulas = particulas
        self.iteraciones = iteraciones
        self.inercia = inercia
        self.cognitivo = cognitivo
        self.social = social
        self.escala_inicial = escala_inicial
        self.velocidad_maxima = velocidad_maxima
        self.semilla = semilla

        self.historial = []
        self.evaluaciones = 0
        self.mejores_parametros = None

    def optimizar(self, modelo, X, y):

        rng = np.random.default_rng(self.semilla)

        # Inicializar el centro de búsqueda
        centro = modelo.parametros_iniciales(
            self.semilla
        )

        # Crear enjambre
        posiciones = centro + rng.normal(
            0,
            self.escala_inicial,
            size=(self.particulas, centro.size)
        )

        posiciones[0] = centro.copy()

        # Inicializar velocidades
        velocidades = np.zeros_like(posiciones)

        # Mejores posiciones individuales
        mejores_personales = posiciones.copy()

        perdidas_personales = np.full(
            self.particulas,
            np.inf
        )

        # Mejor posición global
        mejor_global = posiciones[0].copy()
        perdida_global = np.inf

        self.historial = []
        self.evaluaciones = 0

        print(
            f"\nIniciando PSO: "
            f"{modelo.__class__.__name__}",
            flush=True
        )

        for paso in range(
            1, self.iteraciones + 1
        ):

            # Evaluar cada partícula
            perdidas = np.array([
                modelo.perdida(
                    posicion,
                    X,
                    y
                )
                for posicion in posiciones
            ])

            self.evaluaciones += self.particulas

            if not np.all(np.isfinite(perdidas)):
                raise ValueError(
                    "Se detectaron pérdidas no finitas"
                )

            # Actualizar mejores posiciones personales
            mejora = (
                perdidas < perdidas_personales
            )

            mejores_personales[mejora] = (
                posiciones[mejora]
            )

            perdidas_personales[mejora] = (
                perdidas[mejora]
            )

            # Actualizar mejor posición global
            indice = int(
                np.argmin(perdidas_personales)
            )

            if perdidas_personales[indice] < perdida_global:

                perdida_global = float(
                    perdidas_personales[indice]
                )

                mejor_global = (
                    mejores_personales[indice].copy()
                )

            # Guardar convergencia
            self.historial.append(perdida_global)

            if (
                paso == 1
                or paso % 10 == 0
                or paso == self.iteraciones
            ):
                print(
                    f"Iteración {paso:3d}/"
                    f"{self.iteraciones} | "
                    f"Mejor pérdida: {perdida_global:.6f}",
                    flush=True
                )

            if paso == self.iteraciones:
                break

            # Factores aleatorios
            r1 = rng.random(posiciones.shape)
            r2 = rng.random(posiciones.shape)

            # Actualizar velocidades
            velocidades = (
                self.inercia * velocidades
                + self.cognitivo * r1
                * (mejores_personales - posiciones)
                + self.social * r2
                * (mejor_global - posiciones)
            )

            # Limitar velocidad
            velocidades = np.clip(
                velocidades,
                -self.velocidad_maxima,
                self.velocidad_maxima
            )

            # Actualizar posiciones
            posiciones = posiciones + velocidades

        self.mejores_parametros = (
            mejor_global.copy()
        )

        print(
            f"Optimización finalizada | "
            f"Mejor pérdida: {perdida_global:.6f}"
        )

        print(
            f"Evaluaciones de la función objetivo: "
            f"{self.evaluaciones}"
        )

        return self.mejores_parametros
