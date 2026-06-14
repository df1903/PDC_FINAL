# Risks

## RIESGO-01 — GPU no disponible

**Probabilidad**: media. **Impacto**: alto.

El entorno de desarrollo (Windows) puede no tener GPU CUDA disponible o drivers instalados.

**Mitigación**: implementar `scoring_pycuda.py` con fallback a CPU cuando CUDA no está disponible. Documentar los resultados de speedup obtenidos en el entorno con GPU disponible.

## RIESGO-02 — Compilación MPI en Windows nativo

**Probabilidad**: alta. **Impacto**: medio.

OpenMPI y MPICH tienen soporte limitado en Windows nativo (sin WSL).

**Mitigación**: usar WSL2 o un contenedor Docker para compilar y ejecutar C/MPI. Documentar el entorno de compilación en el informe técnico.

## RIESGO-03 — Divergencia de AUC entre implementaciones

**Probabilidad**: media. **Impacto**: alto.

El cálculo de AUC por conteo de pares (C) puede diferir del `roc_auc_score` de scikit-learn si hay empates o errores de punto flotante acumulados.

**Mitigación**: verificar con dataset pequeño (K=100, N=3) que ambas versiones producen AUC idéntico antes de escalar.

## RIESGO-04 — Datos no separables

**Probabilidad**: baja. **Impacto**: medio.

Como `P = W₁T+W₂S+W₃F` es común a todas las muestras, la separación entre grupos depende enteramente de cómo `A` pondera los ítems por muestra. Si `A` (y los perfiles T/F) no están correlacionados con la etiqueta, ningún W del simplex separa los grupos y AUC ≈ 0.5.

**Mitigación**: `generate_data.py` inyecta señal diferencial en las filas enfermas de `A` (mayor peso a ítems con F/T alto), de modo que un W que enfatice F/T produzca Score más alto en enfermas (ver DEC-08). Verificar tras generar: si AUC_max < 0.6, intensificar la señal y documentar.

## RIESGO-05 — Overhead de MPI supera ganancia con K pequeño

**Probabilidad**: media. **Impacto**: bajo.

Con K=100k y pocos procesos MPI, el overhead de comunicación puede reducir el speedup teórico a valores < 1.

**Mitigación**: medir con K ∈ {100k, 500k, 1M} y reportar curvas de speedup vs K para encontrar el K mínimo amortizable.

## RIESGO-06 — Disponibilidad y límites de sesión de GPU en Colab

**Probabilidad**: media. **Impacto**: medio.

La Fase 4 depende del runtime GPU gratuito de Google Colab, sujeto a cuotas, desconexiones y variabilidad de hardware (T4/otros), lo que afecta la reproducibilidad del speedup de CUDA.

**Mitigación**: fijar `seed=42`, registrar el modelo de GPU usado, repetir la medición y reportar promedios. Conservar `scoring_kernel.cu`/`scoring_pycuda.py` como fuentes para reejecutar. Complementa a RIESGO-01.
