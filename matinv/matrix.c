#define _GNU_SOURCE
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/wait.h>

/* ================= Arena compartida ================= */
static char  *g_base = NULL;
static size_t g_size = 0, g_top = 0;

int arena_init(size_t bytes) {
    if (getenv("ARENA_ANON")) {            /* alternativa si /dev/shm es pequeño */
        g_base = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    } else {
        char name[64];
        snprintf(name, sizeof name, "/matinv_%d", (int)getpid());
        int fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0600);
        if (fd < 0) { perror("shm_open"); return -1; }
        shm_unlink(name);                  /* se libera al terminar el proceso */
        if (ftruncate(fd, (off_t)bytes) < 0) { perror("ftruncate"); close(fd); return -1; }
        g_base = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
    }
    if (g_base == MAP_FAILED) { perror("mmap"); return -1; }
    g_size = bytes; g_top = 0;
    return 0;
}
void arena_destroy(void) { if (g_base) munmap(g_base, g_size); g_base = NULL; }

Mat arena_mat(int r, int c) {
    size_t bytes = ((size_t)r * c * sizeof(double) + 63) & ~(size_t)63;
    if (g_top + bytes > g_size) { fprintf(stderr, "arena llena\n"); exit(2); }
    Mat m = { (double *)(g_base + g_top), r, c, c };
    g_top += bytes;
    return m;
}
size_t arena_mark(void) { return g_top; }
void   arena_release(size_t mark) { g_top = mark; }

/* ================= Utilidades ================= */
void mat_copy(Mat s, Mat d) {
    for (int i = 0; i < s.r; i++) memcpy(at(d, i, 0), at(s, i, 0), (size_t)s.c * sizeof(double));
}
void mat_identity(Mat m) {
    for (int i = 0; i < m.r; i++) {
        memset(at(m, i, 0), 0, (size_t)m.c * sizeof(double));
        if (i < m.c) *at(m, i, i) = 1.0;
    }
}
void mat_diagdom(Mat m, unsigned seed) {
    srand48(seed);
    for (int i = 0; i < m.r; i++) {
        double s = 0;
        for (int j = 0; j < m.c; j++) { double v = drand48() * 2 - 1; *at(m, i, j) = v; s += fabs(v); }
        *at(m, i, i) += (*at(m, i, i) >= 0 ? 1 : -1) * (s + 1.0);  /* |a_ii| > suma resto */
    }
}
double residual_inf(Mat A, Mat Ai) {
    size_t mk = arena_mark();
    Mat R = arena_mat(A.r, A.c);
    Gemm g = { A, Ai, R, 1.0, 0.0 };
    gemm_rows(&g, 0, A.r);
    double best = 0;
    for (int i = 0; i < R.r; i++) {
        double s = 0;
        for (int j = 0; j < R.c; j++) s += fabs(*at(R, i, j) - (i == j ? 1.0 : 0.0));
        if (s > best) best = s;
    }
    arena_release(mk);
    return best;
}

/* Gauss-Jordan con pivoteo parcial (caso base y referencia secuencial) */
int gauss_jordan(Mat M, Mat Out) {
    int n = M.r;
    double *w = malloc((size_t)n * n * sizeof(double));
    if (!w) return -1;
    for (int i = 0; i < n; i++) memcpy(w + (size_t)i * n, at(M, i, 0), n * sizeof(double));
    mat_identity(Out);
    for (int c = 0; c < n; c++) {
        int p = c; double best = fabs(w[(size_t)c * n + c]);
        for (int i = c + 1; i < n; i++) { double v = fabs(w[(size_t)i * n + c]); if (v > best) { best = v; p = i; } }
        if (best < 1e-13) { free(w); return -1; }
        if (p != c) {
            for (int j = 0; j < n; j++) {
                double t = w[(size_t)c * n + j]; w[(size_t)c * n + j] = w[(size_t)p * n + j]; w[(size_t)p * n + j] = t;
                t = *at(Out, c, j); *at(Out, c, j) = *at(Out, p, j); *at(Out, p, j) = t;
            }
        }
        double inv = 1.0 / w[(size_t)c * n + c];
        for (int j = 0; j < n; j++) { w[(size_t)c * n + j] *= inv; *at(Out, c, j) *= inv; }
        for (int i = 0; i < n; i++) {
            if (i == c) continue;
            double f = w[(size_t)i * n + c];
            if (f == 0.0) continue;
            for (int j = 0; j < n; j++) {
                w[(size_t)i * n + j] -= f * w[(size_t)c * n + j];
                *at(Out, i, j)       -= f * *at(Out, c, j);
            }
        }
    }
    free(w);
    return 0;
}

/* ================= Producto de matrices ================= */
#define BK 64
void gemm_rows(const Gemm *g, int r0, int r1) {
    int K = g->A.c, N = g->C.c;
    for (int i = r0; i < r1; i++) {
        double *c = at(g->C, i, 0);
        if (g->beta == 0.0)      memset(c, 0, (size_t)N * sizeof(double));
        else if (g->beta != 1.0) for (int j = 0; j < N; j++) c[j] *= g->beta;
        for (int k0 = 0; k0 < K; k0 += BK) {
            int k1 = k0 + BK < K ? k0 + BK : K;
            for (int k = k0; k < k1; k++) {
                double a = g->alpha * *at(g->A, i, k);
                const double *b = at(g->B, k, 0);
                for (int j = 0; j < N; j++) c[j] += a * b[j];
            }
        }
    }
}

typedef struct { const Gemm *g; int r0, r1; } Slice;
static void *worker(void *arg) { Slice *s = arg; gemm_rows(s->g, s->r0, s->r1); return NULL; }

/* Cada hilo escribe filas distintas de C y solo lee A y B: no hay carreras. */
static void threads_range(const Gemm *g, int r0, int r1, int nt) {
    int rows = r1 - r0;
    if (nt > rows) nt = rows;
    if (nt > 64) nt = 64;
    if (nt <= 1) { gemm_rows(g, r0, r1); return; }
    pthread_t th[64]; Slice sl[64];
    for (int t = 0; t < nt; t++) {
        sl[t].g = g; sl[t].r0 = r0 + (int)((long)rows * t / nt); sl[t].r1 = r0 + (int)((long)rows * (t + 1) / nt);
    }
    for (int t = 1; t < nt; t++)
        if (pthread_create(&th[t], NULL, worker, &sl[t])) { perror("pthread_create"); exit(1); }
    worker(&sl[0]);
    for (int t = 1; t < nt; t++) pthread_join(th[t], NULL);
}
void gemm_threads(const Gemm *g, int nt) { threads_range(g, 0, g->C.r, nt); }

/* Cada proceso hijo calcula un rango de filas; C vive en memoria compartida. */
void gemm_procs(const Gemm *g, int np, int nt) {
    int rows = g->C.r;
    if (np > rows) np = rows;
    if (np <= 1) { threads_range(g, 0, rows, nt); return; }
    pid_t pid[64];
    if (np > 64) np = 64;
    for (int p = 1; p < np; p++) {
        int r0 = (int)((long)rows * p / np), r1 = (int)((long)rows * (p + 1) / np);
        pid[p] = fork();
        if (pid[p] < 0) { perror("fork"); exit(1); }
        if (pid[p] == 0) { threads_range(g, r0, r1, nt); _exit(0); }
    }
    threads_range(g, 0, (int)((long)rows / np), nt);
    for (int p = 1; p < np; p++) {
        int st; waitpid(pid[p], &st, 0);
        if (!WIFEXITED(st) || WEXITSTATUS(st)) { fprintf(stderr, "hijo fallo\n"); exit(1); }
    }
}
