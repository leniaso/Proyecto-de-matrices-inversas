#define _GNU_SOURCE
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

/* Un producto independiente, segun la version. */
static void do_gemm(const Gemm *g, const Cfg *c) {
    switch (c->v) {
    case V_SEQ:     gemm_rows(g, 0, g->C.r); break;
    case V_THREADS: gemm_threads(g, c->P); break;
    case V_PROCS:   gemm_procs(g, c->P, 1); break;
    case V_HIER:    gemm_threads(g, c->P); break;   /* productos dependientes: todos los hilos */
    }
}

/* Dos productos independientes (nodos verdes de la Figura 1).
 * HIER: cada uno en un proceso distinto y, dentro, hilos por filas. */
static void do_pair(const Gemm *a, const Gemm *b, const Cfg *c) {
    if (c->v != V_HIER) { do_gemm(a, c); do_gemm(b, c); return; }
    int t = c->P / 2; if (t < 1) t = 1;
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); exit(1); }
    if (pid == 0) { gemm_threads(b, t); _exit(0); }
    gemm_threads(a, t);
    int st; waitpid(pid, &st, 0);
    if (!WIFEXITED(st) || WEXITSTATUS(st)) { fprintf(stderr, "hijo fallo\n"); exit(1); }
}

static int inv_rec(Mat M, Mat Out, const Cfg *cfg) {
    int n = M.r;
    if (n <= cfg->n0 || n < 2) return gauss_jordan(M, Out);

    int h = n / 2, m = n - h;                  /* funciona para n no potencia de 2 */
    Mat A = sub(M, 0, 0, h, h), B = sub(M, 0, h, h, m);
    Mat C = sub(M, h, 0, m, h), D = sub(M, h, h, m, m);
    Mat O11 = sub(Out, 0, 0, h, h), O12 = sub(Out, 0, h, h, m);
    Mat O21 = sub(Out, h, 0, m, h), O22 = sub(Out, h, h, m, m);

    size_t mk = arena_mark();
    Mat Pm = arena_mat(h, m), Qm = arena_mat(m, h), S = arena_mat(m, m);

    /* 1) X = A^-1 (recursiva, tramo secuencial) -> O11 */
    if (inv_rec(A, O11, cfg)) { arena_release(mk); return -1; }

    /* 2) P = X*B  y  Q = C*X   (independientes) */
    Gemm gP = { O11, B, Pm, 1.0, 0.0 }, gQ = { C, O11, Qm, 1.0, 0.0 };
    do_pair(&gP, &gQ, cfg);

    /* 3) S = D - C*P */
    mat_copy(D, S);
    Gemm gS = { C, Pm, S, -1.0, 1.0 };
    do_gemm(&gS, cfg);

    /* 4) T = S^-1 (recursiva) -> O22 */
    if (inv_rec(S, O22, cfg)) { arena_release(mk); return -1; }

    /* 5) M12 = -P*T  y  M21 = -T*Q   (independientes) */
    Gemm g12 = { Pm, O22, O12, -1.0, 0.0 }, g21 = { O22, Qm, O21, -1.0, 0.0 };
    do_pair(&g12, &g21, cfg);

    /* 6) M11 = X - M12*Q   (P*T*Q = -M12*Q) */
    Gemm g11 = { O12, Qm, O11, -1.0, 1.0 };
    do_gemm(&g11, cfg);

    arena_release(mk);
    return 0;
}

int invert(Mat M, Mat Out, const Cfg *cfg) { return inv_rec(M, Out, cfg); }
