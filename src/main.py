from src.config import CREDITCARD, MAGIC, RAIZ
from src.punto_1.experimentos import generar_experimentos


def main() -> None:
    print(f"Proyecto: {RAIZ}")
    print(f"creditcard.csv: {'encontrado' if CREDITCARD.is_file() else 'pendiente'}")
    print(f"magic.csv: {'encontrado' if MAGIC.is_file() else 'pendiente'}")
    configs = generar_experimentos()
    print(f"Experimentos configurados para el punto 1: {len(configs)}")
    print("Entorno listo. Este comando no entrena modelos.")


if __name__ == "__main__":
    main()
