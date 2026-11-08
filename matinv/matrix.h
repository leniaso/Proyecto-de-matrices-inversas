#ifndef MATRIX_H
#define MATRIX_H
#include <stddef.h>

/* Vista de matriz (row-major). Permite trabajar con bloques sin copiar. */
typedef struct { double *p; int r, c, ld; } Mat;

static inline double *at(Mat m, int i, int j) { return m.p + (size_t)i * m.ld + j; }
static inline Mat sub(Mat m, int r0, int c0, int r, int c) {
    Mat s = { m.p + (size_t)r0 * m.ld + c0, r, c, m.ld };
    return s;
}

/* ---- Arena en memoria compartida (shm_open + mmap) ---- */
int    arena_init(size_t bytes);           /* 0 = ok */
void   arena_destroy(void);
Mat    arena_mat(int r, int c);
size_t arena_mark(void);
void   arena_release(size_t mark);

/* ---- Utilidades ---- */
void   mat_copy(Mat src, Mat dst);
void   mat_identity(Mat m);
void   mat_diagdom(Mat m, unsigned seed);  /* estrictamente diagonal dominante */
double residual_inf(Mat A, Mat Ainv);      /* ||A*Ainv - I||_inf */
int    gauss_jordan(Mat M, Mat Out);       /* caso base; -1 si singular */

/* ---- Producto C = beta*C + alpha*A*B ---- */
typedef struct { Mat A, B, C; double alpha, beta; } Gemm;
void gemm_rows(const Gemm *g, int r0, int r1);          /* secuencial */
void gemm_threads(const Gemm *g, int nthreads);         /* filas entre hilos */
void gemm_procs(const Gemm *g, int nprocs, int nthreads); /* filas entre procesos (y hilos) */

/* ---- Inversión por bloques (complemento de Schur) ---- */
typedef enum { V_SEQ, V_THREADS, V_PROCS, V_HIER } Version;
typedef struct { Version v; int P; int n0; } Cfg;
int invert(Mat M, Mat Out, const Cfg *cfg);             /* 0 ok, -1 singular */

#endif
