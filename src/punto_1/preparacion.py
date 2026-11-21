from sklearn.model_selection import train_test_split

from src.punto_1.datos import cargar_datos


def dividir_datos(test_size=0.20, random_state=42):

    # Cargar las características y las etiquetas
    X, y = cargar_datos()

    # Dividir los datos de manera estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def mostrar_distribucion(y, nombre):

    print(f"\n=== {nombre} ===")
    print(f"Total de registros: {len(y)}")

    print("\nDistribución por clase:")
    print(y.value_counts().sort_index())

    print("\nPorcentajes:")
    print(
        (y.value_counts(normalize=True).sort_index() * 100)
        .round(4)
    )


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = dividir_datos()

    print("=== DIMENSIONES ===")
    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")

    mostrar_distribucion(y_train, "ENTRENAMIENTO")
    mostrar_distribucion(y_test, "PRUEBA")