
import numpy as np

from src.punto_2.preparacion import PreparadorMagic
from src.punto_2.supervisado.modelos.regresion_logistica import (
    RegresionLogisticaMagic
)
from src.punto_2.supervisado.modelos.regresion_neuronal import (
    RegresionNeuronalMagic
)


def main():

    preparador = PreparadorMagic()

    X_train, _, y_train, _ = preparador.preparar()

    # Muestra para comprobar los cálculos
    X = X_train[:500]
    y = y_train.iloc[:500].to_numpy()

    modelos = [
        RegresionLogisticaMagic(),
        RegresionNeuronalMagic()
    ]

    for modelo in modelos:

        print(
            f"\n=== {modelo.__class__.__name__} ==="
        )

        parametros = modelo.parametros_iniciales()

        perdida, gradiente = (
            modelo.perdida_y_gradiente(
                parametros,
                X,
                y
            )
        )

        probabilidades = modelo.probabilidades(
            X,
            parametros
        )

        print(
            f"Parámetros: {modelo.numero_parametros}"
        )

        print(f"Pérdida inicial: {perdida:.6f}")

        print(
            f"Dimensión del gradiente: "
            f"{gradiente.shape}"
        )

        print(
            f"Primeras probabilidades: "
            f"{probabilidades[:5].round(4)}"
        )

        assert np.isfinite(perdida)
        assert np.all(np.isfinite(gradiente))
        assert gradiente.size == modelo.numero_parametros

    print("\nAmbos modelos funcionan correctamente.")


if __name__ == "__main__":
    main()
