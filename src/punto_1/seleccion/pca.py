from sklearn.decomposition import PCA


class SelectorPCA:
    """Fábrica de PCA. El escalado se integra antes, en Pipeline."""

    @staticmethod
    def crear(varianza_explicada: float = 0.95) -> PCA:
        return PCA(n_components=varianza_explicada, svd_solver="full")
