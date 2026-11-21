from src.punto_1.datos import cargar_datos


def explorar_datos():

    # Cargar las características y etiquetas
    X, y = cargar_datos()

    # Información general
    print("=== INFORMACIÓN GENERAL ===")
    print(f"Total de registros: {len(X)}")
    print(f"Total de características: {X.shape[1]}")

    # Distribución de clases
    print("\n=== DISTRIBUCIÓN DE CLASES ===")
    print(y.value_counts())

    # Porcentaje de cada clase
    print("\n=== PORCENTAJES ===")
    print(y.value_counts(normalize=True) * 100)

    # Datos faltantes
    print("\n=== VALORES NULOS ===")
    print(X.isnull().sum().sum())

    # Características disponibles
    print("\n=== COLUMNAS ===")
    print(X.columns.tolist())


if __name__ == "__main__":
    explorar_datos()