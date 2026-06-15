/*
 * npy_io — parser .npy v1.0 minimo (ver npy_io.h).
 *
 * Formato verificado con `xxd` sobre los .npy de `code/data/n_50/`:
 *   offset 0-5  : magic "\x93NUMPY"
 *   offset 6-7  : version (1, 0)
 *   offset 8-9  : header_len (uint16 little-endian)
 *   offset 10.. : header ASCII (dict literal con descr/fortran_order/shape)
 *   offset 10+header_len.. : datos crudos en C-order
 */
#include "npy_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NPY_MAGIC_LEN 6
static const unsigned char NPY_MAGIC[NPY_MAGIC_LEN] = {0x93, 'N', 'U', 'M', 'P', 'Y'};

/* Lee magic + version + header_len y devuelve el header ASCII (malloc'd,
 * terminado en '\0'). Termina el programa en caso de error de formato. */
static char *read_header(FILE *f, const char *path) {
    unsigned char magic[NPY_MAGIC_LEN];
    unsigned char version[2];
    unsigned char len_bytes[2];

    if (fread(magic, 1, NPY_MAGIC_LEN, f) != NPY_MAGIC_LEN ||
        memcmp(magic, NPY_MAGIC, NPY_MAGIC_LEN) != 0) {
        fprintf(stderr, "npy_io: %s: magic invalido (no es .npy)\n", path);
        exit(1);
    }
    if (fread(version, 1, 2, f) != 2 || version[0] != 1 || version[1] != 0) {
        fprintf(stderr, "npy_io: %s: version .npy no soportada (esperado 1.0)\n", path);
        exit(1);
    }
    if (fread(len_bytes, 1, 2, f) != 2) {
        fprintf(stderr, "npy_io: %s: header_len truncado\n", path);
        exit(1);
    }

    int header_len = (int)len_bytes[0] | ((int)len_bytes[1] << 8);
    char *header = malloc((size_t)header_len + 1);
    if (!header) {
        fprintf(stderr, "npy_io: %s: sin memoria para header\n", path);
        exit(1);
    }
    if (fread(header, 1, (size_t)header_len, f) != (size_t)header_len) {
        fprintf(stderr, "npy_io: %s: header truncado\n", path);
        exit(1);
    }
    header[header_len] = '\0';
    return header;
}

/* Extrae `shape` del header (tuplas "(a, b)" o "(a,)"). */
static void parse_shape(const char *header, const char *path, int *dim0, int *dim1) {
    const char *p = strstr(header, "'shape': (");
    if (!p) {
        fprintf(stderr, "npy_io: %s: 'shape' no encontrado en header\n", path);
        exit(1);
    }
    p += strlen("'shape': (");

    char *end;
    long d0 = strtol(p, &end, 10);
    p = end;
    while (*p == ' ' || *p == ',') p++;
    if (*p == ')') {
        *dim0 = (int)d0;
        *dim1 = 1;
        return;
    }
    long d1 = strtol(p, &end, 10);
    *dim0 = (int)d0;
    *dim1 = (int)d1;
}

/* Carga generica: valida `descr`/`fortran_order`, lee shape y datos crudos. */
static void *npy_load(const char *path, const char *expected_descr, size_t elem_size,
                       int *rows, int *cols) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "npy_io: no se pudo abrir %s\n", path);
        exit(1);
    }

    char *header = read_header(f, path);

    if (!strstr(header, expected_descr)) {
        fprintf(stderr, "npy_io: %s: descr distinto de %s\n", path, expected_descr);
        exit(1);
    }
    if (!strstr(header, "'fortran_order': False")) {
        fprintf(stderr, "npy_io: %s: se esperaba fortran_order=False\n", path);
        exit(1);
    }

    int dim0, dim1;
    parse_shape(header, path, &dim0, &dim1);
    free(header);

    size_t count = (size_t)dim0 * (size_t)dim1;
    void *data = malloc(count * elem_size);
    if (!data) {
        fprintf(stderr, "npy_io: %s: sin memoria para %zu elementos\n", path, count);
        exit(1);
    }
    if (fread(data, elem_size, count, f) != count) {
        fprintf(stderr, "npy_io: %s: datos truncados\n", path);
        free(data);
        exit(1);
    }
    fclose(f);

    *rows = dim0;
    *cols = dim1;
    return data;
}

float *npy_load_f32(const char *path, int *rows, int *cols) {
    return (float *)npy_load(path, "'descr': '<f4'", sizeof(float), rows, cols);
}

int32_t *npy_load_i32(const char *path, int *rows, int *cols) {
    return (int32_t *)npy_load(path, "'descr': '<i4'", sizeof(int32_t), rows, cols);
}
