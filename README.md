# Proyecto de Machine Learning — Clasificación y optimización

Proyecto académico de **aprendizaje de máquinas**, desarrollado en Python. Incluye dos estudios: detección de fraude con datos desbalanceados y análisis del conjunto **MAGIC Gamma Telescope** mediante optimización supervisada y agrupamiento no supervisado.

El proyecto organiza el código en módulos reutilizables (`src/`), presenta los experimentos y sus interpretaciones en cuatro notebooks (`notebooks/`) y conserva las métricas y figuras en `resultados/`.

## Requisitos

- **Python 3.11** (versión utilizada durante el desarrollo: **3.11.9**).
- Visual Studio Code con las extensiones **Python** y **Jupyter** (recomendado).
- Git, si deseas clonar o contribuir al repositorio.

## Instalación en Windows / PowerShell

Desde la raíz del proyecto:

```powershell
# Crear y activar el entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Instalar dependencias
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Para ejecutar los notebooks en VS Code, si no están instalados:
python -m pip install ipykernel jupyter

# Verificaciones iniciales
python -m src.main
python -m pip check
python -m pytest -q
```

Selecciona como **kernel** de los notebooks el intérprete `.venv\Scripts\python.exe`.

> Si PowerShell impide activar el entorno virtual, puedes invocar directamente `.\.venv\Scripts\python.exe` en lugar de `python`, o consultar la política de ejecución de tu equipo.

## Conjuntos de datos

**Los datasets originales no se incluyen en GitHub**. Los archivos del proyecto deben obtenerse de las fuentes suministradas para el curso, respetando sus condiciones de uso.

| Punto | Conjunto de datos | Archivo necesario | Descripción |
|---|---|---|---|
| 1 | Credit Card Fraud Detection | `creditcard.csv` | Transacciones legítimas y fraudulentas; variable objetivo `Class` (0/1) |
| 2 | MAGIC Gamma Telescope | `magic.csv` | Eventos gamma (`g`) y hadrónicos (`h`) |
| 2 | Documentación de MAGIC | `magic.rst` | Descripción de atributos del conjunto de datos |

Una vez descargados, coloca los archivos en esta estructura:

```text
data/
├── raw/
│   ├── creditcard.csv
│   ├── magic.csv
│   └── magic.rst
└── processed/
```

**Importante:** el cargador de MAGIC del proyecto espera la versión de `magic.csv` utilizada en el taller: una primera fila de metadatos, una segunda fila con encabezados y, después, los registros. Si descargas otra distribución de MAGIC, revisa su formato antes de utilizar `src/punto_2/datos.py`.

El archivo `creditcard.csv` ocupa aproximadamente **143,84 MB**, por lo que se excluye del repositorio junto con los demás archivos de datos originales. La carpeta `data/raw/` debe crearse si no aparece después de clonar el proyecto.

## Organización del proyecto

```text
Proyecto_Machine_Learning/
├── data/
│   ├── raw/                  # Datasets locales (ignorados por Git)
│   └── processed/            # Archivos de procesamiento
├── notebooks/
│   ├── 01_exploracion_fraude.ipynb
│   ├── 02_modelos_fraude.ipynb
│   ├── 03_optimizacion_magic.ipynb
│   └── 04_agrupamiento_magic.ipynb
├── src/
│   ├── comun/                # Métricas reutilizables
│   ├── punto_1/              # Fraude: selección, modelos y evaluación
│   ├── punto_2/              # MAGIC: optimizadores y agrupamiento
│   ├── config.py
│   └── main.py
├── resultados/
│   ├── punto_1/
│   └── punto_2/
├── tests/
├── requirements.txt
└── README.md
```

## Punto 1 — Clasificación de fraude

Se evaluaron **32 configuraciones** obtenidas al combinar cuatro familias de clasificadores (regresión logística, árbol de decisión, SVM y red neuronal), dos estrategias de características (correlación y PCA) y cuatro variantes por familia. La selección se basó en validación cruzada estratificada y **F1-score macro**.

### Comandos de ejecución

```powershell
# 1(i): Exploración y organización
python -m src.punto_1.exploracion
python -m src.punto_1.preparacion

# 1(ii): Selección y transformación de características
python -m src.punto_1.analizar_seleccion

# Validación inicial
python -m src.punto_1.validacion_inicial
python -m src.punto_1.graficar_validacion_inicial

# 1(iii): Validación cruzada de las 32 configuraciones
python -m src.punto_1.validacion_cruzada --modelo logistica
python -m src.punto_1.validacion_cruzada --modelo arbol
python -m src.punto_1.validacion_cruzada --modelo svm
python -m src.punto_1.validacion_cruzada --modelo red_neuronal

# 1(iv): Análisis del Top 5 y evaluación final
python -m src.punto_1.analizar_top5
python -m src.punto_1.evaluacion_final
```

**Resultado destacado:** `red_neuronal_pca_02` obtuvo F1 macro promedio de **0,9174** en validación cruzada y **0,9085** en la evaluación de prueba. El conjunto está muy desbalanceado: aproximadamente el **0,173 %** de las transacciones corresponde a fraude. Por eso también se analizan la sensibilidad, la precisión y las matrices de confusión.

## Punto 2(i) — MAGIC: clasificación y optimización

Se implementaron regresión logística y una red neuronal pequeña, entrenadas mediante **descenso por gradiente, algoritmo genético (GA), Particle Swarm Optimization (PSO) y Simulated Annealing (SA)**. Se compararon las ocho combinaciones, se repitieron los experimentos con tres semillas y se realizó una evaluación final separada.

```powershell
# Exploración y preparación
python -m src.punto_2.exploracion
python -m src.punto_2.preparacion

# Comprobación de modelos
python -m src.punto_2.verificar_modelos

# Entrenamiento con cuatro optimizadores
python -m src.punto_2.supervisado.verificar_modelos
python -m src.punto_2.supervisado.entrenar_gradiente
python -m src.punto_2.supervisado.entrenar_genetico
python -m src.punto_2.supervisado.entrenar_pso
python -m src.punto_2.supervisado.entrenar_sa

# Comparación y estabilidad
python -m src.punto_2.supervisado.comparar_optimizadores
python -m src.punto_2.supervisado.validacion_semillas

# Evaluación final
python -m src.punto_2.supervisado.evaluacion_final
```

**Resultado destacado:** la regresión neuronal optimizada con PSO alcanzó **F1-score de 0,8546** y **exactitud de 80,28 %** en los 3.804 registros de prueba. La regresión logística optimizada con GA obtuvo F1-score de **0,8386**. Los resultados corresponden a los experimentos y condiciones de evaluación documentados; no representan superioridad general de un algoritmo.

## Punto 2(ii) — MAGIC: aprendizaje no supervisado

Se compararon tres métodos sobre una muestra de 3.000 registros, con características estandarizadas y sin emplear las etiquetas reales durante el agrupamiento:

```powershell
python -m src.punto_2.no_supervisado.kmeans
python -m src.punto_2.no_supervisado.jerarquico
python -m src.punto_2.no_supervisado.dbscan
python -m src.punto_2.no_supervisado.comparar_agrupamientos
```

| Método | Clusters seleccionados | Silhouette | ARI externo |
|---|---:|---:|---:|
| K-Means | 2 | 0,2855 | 0,0039 |
| Jerárquico Ward | 2 | 0,3193 | 0,0450 |
| DBSCAN | 2 | 0,5450 | 0,0516 |

**Precaución al interpretar DBSCAN:** identificó un cluster de **2.900** observaciones y otro de solo **5**, además de **95** registros de ruido. Su Silhouette se calculó excluyendo el ruido, por lo que no es directamente equivalente a los otros métodos. Los ARI cercanos a cero indican que los clusters encontrados no reproducen claramente las clases gamma/hadrón.

## Notebooks: explicación y visualización

Los notebooks presentan el procedimiento, las tablas, las gráficas, los resultados y sus interpretaciones, sin copiar toda la implementación de `src/`:

| Notebook | Contenido |
|---|---|
| `01_exploracion_fraude.ipynb` | Exploración, desbalance y división estratificada de fraude |
| `02_modelos_fraude.ipynb` | Selección de características, 32 modelos, Top 5 y evaluación |
| `03_optimizacion_magic.ipynb` | Comparación de optimizadores, estabilidad y evaluación supervisada |
| `04_agrupamiento_magic.ipynb` | K-Means, Ward, DBSCAN y discusión no supervisada |

Para usarlos, ábrelos en VS Code, selecciona el kernel `.venv` y ejecuta las celdas de arriba hacia abajo. **Las celdas que leen los datasets requieren los archivos locales en `data/raw/`.** Las celdas que leen resultados también requieren los CSV y figuras correspondientes en `resultados/`. Si los notebooks tienen salidas guardadas, GitHub puede mostrarlas sin ejecutar código.

## Resultados y reproducibilidad

- `resultados/punto_1/`: métricas, tablas y figuras del experimento de fraude.
- `resultados/punto_2/`: resultados de gradiente, GA, PSO, SA, validación con semillas, prueba final y agrupamiento.
- Los archivos en `resultados/` permiten inspeccionar experimentos previos sin repetir todos los entrenamientos.
- Algunos scripts conservan resultados existentes y omiten ejecuciones ya registradas; revisa los CSV antes de intentar repetir un experimento desde cero.

> **Nota:** ejecutar todas las configuraciones y métodos puede requerir recursos computacionales considerables. Para una evaluación metodológicamente correcta, no deben ajustarse hiperparámetros utilizando el conjunto de prueba final.

## Comprobaciones antes de subir a GitHub

Los archivos de datos originales, el entorno virtual y las credenciales deben permanecer fuera de Git. Verifica el estado del repositorio:

```powershell
git check-ignore -v data/raw/creditcard.csv
git check-ignore -v data/raw/magic.csv
git check-ignore -v data/raw/magic.rst

git status
```

Si algún dataset ya estaba siendo rastreado, retíralo únicamente del índice (sin borrarlo del equipo) con `git rm --cached --ignore-unmatch <ruta>`. Si fue incluido en commits previos, también se debe comprobar el historial antes de publicarlo.

---

*Proyecto académico: los resultados se reportan bajo las configuraciones experimentales descritas y no deben interpretarse como garantías de desempeño en producción.*
