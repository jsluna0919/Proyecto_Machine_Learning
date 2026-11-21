
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.punto_2.datos import DatosMagic


class PreparadorMagic:
    """
    Clase encargada de preparar la base de datos
    MAGIC para los modelos de clasificación.
    """

    def __init__(self, test_size=0.20, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.escalador = StandardScaler()

    def dividir_datos(self):
        """
        Divide los datos en entrenamiento y prueba
        manteniendo la proporción de ambas clases.
        """

        datos = DatosMagic()
        X, y = datos.cargar()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        return X_train, X_test, y_train, y_test

    def estandarizar(self, X_train, X_test):
        """
        Ajusta el escalador únicamente con los
        datos de entrenamiento.
        """

        X_train_escalado = self.escalador.fit_transform(
            X_train
        )

        X_test_escalado = self.escalador.transform(
            X_test
        )

        return X_train_escalado, X_test_escalado

    def preparar(self):
        """
        Ejecuta la división y la estandarización.
        """

        X_train, X_test, y_train, y_test = (
            self.dividir_datos()
        )

        X_train_escalado, X_test_escalado = (
            self.estandarizar(X_train, X_test)
        )

        return (
            X_train_escalado,
            X_test_escalado,
            y_train,
            y_test
        )


def main():

    preparador = PreparadorMagic()

    X_train, X_test, y_train, y_test = (
        preparador.preparar()
    )

    print("\n=== DIMENSIONES ===")

    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")

    print("\n=== ENTRENAMIENTO ===")
    print(y_train.value_counts().sort_index())

    print("\n=== PRUEBA ===")
    print(y_test.value_counts().sort_index())

    print("\n=== PORCENTAJES ENTRENAMIENTO ===")
    print(
        (
            y_train.value_counts(normalize=True)
            .sort_index() * 100
        ).round(4)
    )

    print("\n=== VERIFICACIÓN ESCALADO ===")
    print(
        "Media primera característica:",
        round(X_train[:, 0].mean(), 6)
    )

    print(
        "Desviación primera característica:",
        round(X_train[:, 0].std(), 6)
    )


if __name__ == "__main__":
    main()
