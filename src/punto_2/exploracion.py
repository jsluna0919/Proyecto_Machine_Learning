from src.punto_2.datos import DatosMagic


def explorar_datos():

    # Cargar datos
    datos = DatosMagic()
    X, y = datos.cargar()

    print("\n=== BASE DE DATOS MAGIC ===")

    print(f"Total de registros: {len(X)}")
    print(f"Características: {X.shape[1]}")

    print("\n=== COLUMNAS ===")
    print(X.columns.tolist())

    print("\n=== PRIMEROS REGISTROS ===")
    print(X.head())

    print("\n=== DISTRIBUCIÓN DE CLASES ===")
    print(y.value_counts().sort_index())

    print("\n=== PORCENTAJES ===")
    print(
        (y.value_counts(normalize=True).sort_index() * 100)
        .round(4)
    )

    print("\n=== VALORES NULOS ===")
    print(X.isnull().sum().sum())

    print("\n=== ESTADÍSTICAS ===")
    print(X.describe().round(3))


if __name__ == "__main__":
    explorar_datos()