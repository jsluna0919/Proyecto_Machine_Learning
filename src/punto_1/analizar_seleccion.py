import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.punto_1.datos import cargar_datos, dividir_datos
from src.punto_1.seleccion.correlacion import SelectorCorrelacion
from src.punto_1.seleccion.pca import SelectorPCA


def analizar_seleccion():

    # 1. Cargar y dividir los datos
    X, y = cargar_datos()
    X_train, X_test, y_train, y_test = dividir_datos(X, y)

    # 2. Estandarizar solamente con entrenamiento
    escalador = StandardScaler()
    X_train_escalado = escalador.fit_transform(X_train)

    # 3. Selección mediante correlación
    print("\n=== FILTRO DE CORRELACIÓN ===")

    selector = SelectorCorrelacion(k=20)
    selector.fit(X_train_escalado, y_train)

    importancia = pd.Series(
        selector.importancias_,
        index=X_train.columns
    ).sort_values(ascending=False)

    print("\n10 características más correlacionadas con Class:")
    print(importancia.head(10).round(4))

    print("\nCaracterísticas seleccionadas:")
    print(X_train.columns[selector.indices_].tolist())

    # 4. Reducción mediante PCA
    print("\n=== ANÁLISIS PCA ===")

    pca = SelectorPCA.crear(varianza_explicada=0.95)
    pca.fit(X_train_escalado)

    print(f"Características originales: {X_train.shape[1]}")
    print(f"Componentes seleccionados: {pca.n_components_}")

    print(
        f"Varianza explicada: "
        f"{pca.explained_variance_ratio_.sum() * 100:.2f}%"
    )


if __name__ == "__main__":
    analizar_seleccion()