/*
 * npy_io — parser .npy v1.0 minimo (sin librerias externas).
 *
 * Solo lectura de arreglos `<f4` (float32) y `<i4` (int32) en C-order
 * (fortran_order=False), tal como los genera `code/data/generate_data.py`
 * (DEC-06/DEC-10). No modifica los archivos .npy.
 */
#ifndef NPY_IO_H
#define NPY_IO_H

#include <stdint.h>

/* Carga un arreglo .npy float32. `rows`/`cols` reciben el shape
 * (`cols=1` para arreglos 1-D). Termina el programa con exit(1) si el
 * archivo no existe o el formato no coincide con lo esperado. */
float *npy_load_f32(const char *path, int *rows, int *cols);

/* Carga un arreglo .npy int32. `rows`/`cols` reciben el shape
 * (`cols=1` para arreglos 1-D). */
int32_t *npy_load_i32(const char *path, int *rows, int *cols);

#endif /* NPY_IO_H */
