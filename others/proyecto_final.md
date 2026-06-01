# Optimizacion Paralela del Sistema de Scoring Metagenomico para Clasificacion Binaria

*Proyecto de Computacion de Alto Rendimiento (HPC)*

`Python` · `C/OpenMP` · `MPI` · `CUDA`

**Area:** Bioinformatica Computacional  
**Fecha:** 7 de mayo de 2026

---

## 0. Índice

1. [Descripcion del Problema](#1-descripcion-del-problema)
2. [Modelo Matematico](#2-modelo-matematico)
   - 2.1. [Score por Item](#21-score-por-item)
   - 2.2. [Score por Muestra](#22-score-por-muestra)
   - 2.3. [Funcion Objetivo](#23-funcion-objetivo)
   - 2.4. [Validacion del Scoring](#24-validacion-del-scoring)
   - 2.5. [Espacio de Busqueda](#25-espacio-de-busqueda)
3. [Arquitectura del Sistema](#3-arquitectura-del-sistema)
   - 3.1. [Nivel 1 — Python: Baseline](#31-nivel-1--python-baseline)
   - 3.2. [Nivel 2 — C con OpenMP y MPI](#32-nivel-2--c-con-openmp-y-mpi)
     - 3.2.1. [C + OpenMP (Memoria Compartida)](#321-c--openmp-memoria-compartida)
     - 3.2.2. [C + MPI (Memoria Distribuida)](#322-c--mpi-memoria-distribuida)
   - 3.3. [Nivel 3 — CUDA: Aceleracion GPU](#33-nivel-3--cuda-aceleracion-gpu)
4. [Metricas de Evaluacion](#4-metricas-de-evaluacion)
   - 4.1. [Tiempo de Ejecucion](#41-tiempo-de-ejecucion)
   - 4.2. [Speedup](#42-speedup)
   - 4.3. [Eficiencia](#43-eficiencia)
   - 4.4. [Ley de Amdahl](#44-ley-de-amdahl)
   - 4.5. [Tabla Comparativa](#45-tabla-comparativa)
5. [Estructura del Repositorio](#5-estructura-del-repositorio)
6. [Script de Generacion de Datos](#6-script-de-generacion-de-datos)
7. [Entregables](#7-entregables)

---

## 1. Descripcion del Problema

Se busca desarrollar un sistema de optimizacion de alto rendimiento para determinar un vector de pesos **W** que maximice el area bajo la curva ROC (*AUC*) en la clasificacion binaria de muestras biologicas de tipo metagenomico.

El conjunto de datos consiste en **10 muestras** divididas simetricamente:

- **5 muestras** — poblacion *sana* (y = 0).
- **5 muestras** — poblacion *enferma* (y = 1).

Un scoring bien calibrado producira una distribucion de scores donde los dos grupos sean claramente distinguibles. Si la mayoria de scores altos se concentran en las muestras enfermas (o viceversa de modo consistente), el modelo es valido; en caso contrario, la combinacion de pesos **W** no representa adecuadamente los datos.

---

## 2. Modelo Matematico

Cada muestra *j* se describe mediante *N* items (genomas o taxones). Para cada item *i* se dispone de tres perfiles:

**Cuadro 1: Perfiles por item**

| Simbolo | Perfil                | Descripcion                                                          |
|---------|-----------------------|----------------------------------------------------------------------|
| *T_i*   | Taxonomico            | Abundancia relativa de microorganismos.                              |
| *S_i*   | Ecologico poblacional | Variables contextuales no genomicas de la muestra.                   |
| *F_i*   | Funcional             | Presencia/ausencia de genes de interes (beneficos, de resistencia, etc.). |

### 2.1. Score por Item

$$P_i = W_1\, T_i + W_2\, S_i + W_3\, F_i \tag{1}$$

donde **W** = (W₁, W₂, W₃)ᵀ es el vector de pesos a optimizar.

### 2.2. Score por Muestra

$$\mathbf{Score} = A \cdot \mathbf{P} \tag{2}$$

- *A* ∈ ℝ^{10×N}: **matriz de contribucion** (dimensiones fijas). La entrada *a_{ji}* pondera el item *i* en la muestra *j*.
- **P** ∈ ℝ^N: vector de scores por item (ec. 1).
- **Score** ∈ ℝ^{10}: un score escalar por muestra.

### 2.3. Funcion Objetivo

$$\max_{\mathbf{W}}\; \text{AUC}(\mathbf{y},\, \mathbf{Score}(\mathbf{W})) \tag{3}$$

El AUC mide la probabilidad de que una muestra enferma reciba score mayor que una sana; AUC ∈ [0.5, 1.0] es el rango de interes practico.

### 2.4. Validacion del Scoring

Para un umbral de decision θ:

$$\text{Consistencia} = \frac{|\{j : \text{Score}_j > \theta,\; y_j = 1\}|}{5} + \frac{|\{j : \text{Score}_j \leq \theta,\; y_j = 0\}|}{5} \tag{4}$$

Se considera satisfactorio Consistencia ≥ 0.8.

### 2.5. Espacio de Busqueda

$$W_1 + W_2 + W_3 = 1, \qquad W_i \geq 0 \tag{5}$$

---

## 3. Arquitectura del Sistema

**Cuadro 2: Niveles de implementacion**

| Nivel | Tecnologia                          | Paradigma                      | Proposito             |
|-------|-------------------------------------|--------------------------------|-----------------------|
| 1     | Python (NumPy / multiprocessing)    | Secuencial / Multicore         | Baseline y validacion |
| 2     | C + OpenMP · C + MPI                | Mem. compartida / Distribuida  | Paralelismo de CPU    |
| 3     | PyCUDA / CUDA C                     | GPU (SIMD masivo)              | Aceleracion maxima    |

### 3.1. Nivel 1 — Python: Baseline

**Librerias:** `NumPy`, `scikit-learn`, `multiprocessing`.

**Estrategia:** *Random Search* sobre K candidatos de **W** en el simplex.

**Flujo:**

1. Cargar *A* y **y**.
2. Muestrear K vectores **W** sobre el simplex.
3. Para cada **W**: **P** → **Score** = A**P** → AUC.
4. Retornar **W**\* = arg max AUC.
5. Version multicore: repartir los K candidatos entre `cpu_count()` procesos.

### 3.2. Nivel 2 — C con OpenMP y MPI

#### 3.2.1. C + OpenMP (Memoria Compartida)

- Multiplicacion *A* · **P** con `#pragma omp parallel for`.
- Actualizacion del AUC maximo con `#pragma omp critical`.
- Reduccion de acumuladores con clausula `reduction(+:acum)`.

#### 3.2.2. C + MPI (Memoria Distribuida)

- Proceso *root* genera los K candidatos y distribuye con `MPI_Scatter`.
- Cada proceso evalua su subconjunto de forma independiente.
- Recoleccion del optimo con `MPI_Reduce` (op. `MPI_MAX`).

### 3.3. Nivel 3 — CUDA: Aceleracion GPU

- Cada hilo evalua un candidato **W**_k.
- Kernel usa **memoria compartida** para cachear filas de *A* por bloque.
- Transferencias Host-to-Device realizadas una unica vez antes del kernel.
- Reduccion del AUC con *reduction kernel* estandar.

$$\text{Grid} = \left\lceil \frac{K}{\text{BLOCK\_SIZE}} \right\rceil, \qquad \text{BLOCK\_SIZE} = 256 \tag{6}$$

---

## 4. Metricas de Evaluacion

### 4.1. Tiempo de Ejecucion

$$T = t_{\text{fin}} - t_{\text{inicio}} \tag{7}$$

Medido desde el inicio de la busqueda hasta obtener **W**\* (excluye carga de datos).

### 4.2. Speedup

$$S = \frac{T_{\text{Python}}}{T_{\text{impl}}} \tag{8}$$

### 4.3. Eficiencia

$$E = \frac{S}{P} \tag{9}$$

donde *P* es el numero de nucleos/hilos. *E* = 1 indica escalabilidad ideal.

### 4.4. Ley de Amdahl

$$S_{\text{max}} = \frac{1}{(1 - f) + \dfrac{f}{P}} \tag{10}$$

*f* se estima empiricamente como la fraccion del tiempo total que ocupa la parte paralela.

### 4.5. Tabla Comparativa

**Cuadro 3: Resumen de desempenyo (completar con resultados)**

| Implementacion    | T (s) |   S  |  E  | AUC |
|-------------------|-------|------|-----|-----|
| Python secuencial |  —    | 1.00 |  —  |  —  |
| Python multicore  |  —    |  —   |  —  |  —  |
| C + OpenMP        |  —    |  —   |  —  |  —  |
| C + MPI           |  —    |  —   |  —  |  —  |
| CUDA              |  —    |  —   |  —  |  —  |

---

## 5. Estructura del Repositorio

*Listing 1: Organizacion de carpetas*

```
scoring_metagenomico/
|-- data/
|   |-- generate_data.py
|   |-- matrix_A.npy
|   '-- labels.npy
|-- python/
|   |-- sequential.py
|   '-- multicore.py
|-- C_OpenMP_MPI/
|   |-- scoring_openmp.c
|   |-- scoring_mpi.c
|   '-- Makefile
|-- CUDA/
|   |-- scoring_kernel.cu
|   |-- scoring_pycuda.py
|   '-- Makefile
|-- results/
|   |-- benchmark.csv
|   '-- plots/
'-- report/
    '-- informe_tecnico.pdf
```

---

## 6. Script de Generacion de Datos

*Listing 2: data/generate\_data.py*

```python
import numpy as np

def generate_data(n_items: int = 50, seed: int = 42):
    """
    Genera A in R^{10 x n_items} y etiquetas y.
    Filas 0-4: sanas (y=0). Filas 5-9: enfermas (y=1).
    Cada fila de A es Dirichlet(1,...,1) -> suma 1.
    """
    rng = np.random.default_rng(seed)
    A = rng.dirichlet(np.ones(n_items), size=10).astype(np.float32)
    y = np.array([0]*5 + [1]*5, dtype=np.int32)
    return A, y

if __name__ == "__main__":
    A, y = generate_data(n_items=50)
    np.save("data/matrix_A.npy", A)
    np.save("data/labels.npy",   y)
    print(f"A: {A.shape}  |  y: {y}")
```

---

## 7. Entregables

1. **Codigo fuente** organizado en `python/`, `C_OpenMP_MPI/` y `CUDA/`.
2. **Script de generacion de datos** con N configurable para pruebas de escalabilidad.
3. **Script de benchmark** (`run_all.sh`) que genera `results/benchmark.csv`.
4. **Informe tecnico** en PDF con:
   - Estrategias de sincronizacion por nivel.
   - Analisis de gestion de memoria.
   - Graficas comparativas de Speedup y Eficiencia vs. *P*.
   - Analisis de Amdahl con *f* empirico.
   - Discusion de separabilidad de grupos mediante el score optimo.