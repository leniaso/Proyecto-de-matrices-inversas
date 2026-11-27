#define _GNU_SOURCE
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/wait.h>

typedef struct ThreadPool ThreadPool;
typedef struct { ThreadPool *pool; int tid; } PoolArg;

struct ThreadPool {
    int P;
    int stop;
    const Gemm *cur_g;
    pthread_barrier_t bar_start;
    pthread_barrier_t bar_done;
    pthread_t *th;
    PoolArg *args;
};

static void *pool_worker(void *arg) {
    PoolArg *a = arg;
    ThreadPool *pool = a->pool;
    int tid = a->tid;
    for (;;) {
        pthread_barrier_wait(&pool->bar_start);
        if (pool->stop) break;
        const Gemm *g = pool->cur_g;
        int rows = g->C.r;
        int r0 = (int)((long)rows * tid / pool->P);
        int r1 = (int)((long)rows * (tid + 1) / pool->P);
        if (r0 < r1) gemm_rows(g, r0, r1);
        pthread_barrier_wait(&pool->bar_done);
    }
    return NULL;
}

static void pool_init(ThreadPool *pool, int P) {
    pool->P = P < 1 ? 1 : P;
    pool->stop = 0;
    pool->cur_g = NULL;
    pool->th = NULL;
    pool->args = NULL;
    if (pool->P <= 1) return;
    pool->th = malloc((size_t)pool->P * sizeof(pthread_t));
    pool->args = malloc((size_t)pool->P * sizeof(PoolArg));
    if (!pool->th || !pool->args) { perror("malloc"); exit(1); }
    if (pthread_barrier_init(&pool->bar_start, NULL, (unsigned)pool->P) ||
        pthread_barrier_init(&pool->bar_done, NULL, (unsigned)pool->P)) {
        perror("pthread_barrier_init"); exit(1);
    }
    for (int t = 1; t < pool->P; t++) {
        pool->args[t].pool = pool;
        pool->args[t].tid = t;
        if (pthread_create(&pool->th[t], NULL, pool_worker, &pool->args[t])) {
            perror("pthread_create"); exit(1);
        }
    }
}

static void pool_gemm(ThreadPool *pool, const Gemm *g) {
    if (pool->P <= 1) { gemm_rows(g, 0, g->C.r); return; }
    pool->cur_g = g;
    pthread_barrier_wait(&pool->bar_start);
    int rows = g->C.r;
    int r1 = (int)((long)rows / pool->P);
    if (0 < r1) gemm_rows(g, 0, r1);
    pthread_barrier_wait(&pool->bar_done);
}

static void pool_destroy(ThreadPool *pool) {
    if (pool->P <= 1) return;
    pool->stop = 1;
    pthread_barrier_wait(&pool->bar_start);
    for (int t = 1; t < pool->P; t++) pthread_join(pool->th[t], NULL);
    pthread_barrier_destroy(&pool->bar_start);
    pthread_barrier_destroy(&pool->bar_done);
    free(pool->th);
    free(pool->args);
}

/* Un producto independiente, segun la version. */
static void do_gemm(const Gemm *g, const Cfg *c, ThreadPool *pool) {
    switch (c->v) {
    case V_SEQ:             gemm_rows(g, 0, g->C.r); break;
    case V_THREADS:         gemm_threads(g, c->P); break;
    case V_PROCS:           gemm_procs(g, c->P, 1); break;
    case V_HIER:            gemm_threads(g, c->P); break;   /* productos dependientes: todos los hilos */
    case V_PAIRT:           gemm_threads(g, c->P); break;
    case V_THREADS_BARRIER: pool_gemm(pool, g); break;
    }
}

/* Variante de diagnostico PAIRT: igual que HIER pero el par corre en hilos (sin fork). */
typedef struct { const Gemm *g; int t; } PairArg;
static void *pair_worker(void *p) { PairArg *a = p; gemm_threads(a->g, a->t); return NULL; }

/* Dos productos independientes (nodos verdes de la Figura 1).
 * HIER: cada uno en un proceso distinto y, dentro, hilos por filas. */
static void do_pair(const Gemm *a, const Gemm *b, const Cfg *c, ThreadPool *pool) {
    if (c->v == V_PAIRT) {
        int t = c->P / 2; if (t < 1) t = 1;
        PairArg arg = { b, t }; pthread_t th;
        if (pthread_create(&th, NULL, pair_worker, &arg)) { perror("pthread_create"); exit(1); }
        gemm_threads(a, t);
        pthread_join(th, NULL);
        return;
    }
    if (c->v != V_HIER) { do_gemm(a, c, pool); do_gemm(b, c, pool); return; }
    int t = c->P / 2; if (t < 1) t = 1;
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); exit(1); }
    if (pid == 0) { gemm_threads(b, t); _exit(0); }
    gemm_threads(a, t);
    int st; waitpid(pid, &st, 0);
    if (!WIFEXITED(st) || WEXITSTATUS(st)) { fprintf(stderr, "hijo fallo\n"); exit(1); }
}

static int inv_rec(Mat M, Mat Out, const Cfg *cfg, ThreadPool *pool) {
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
    if (inv_rec(A, O11, cfg, pool)) { arena_release(mk); return -1; }

    /* 2) P = X*B  y  Q = C*X   (independientes) */
    Gemm gP = { O11, B, Pm, 1.0, 0.0 }, gQ = { C, O11, Qm, 1.0, 0.0 };
    do_pair(&gP, &gQ, cfg, pool);

    /* 3) S = D - C*P */
    mat_copy(D, S);
    Gemm gS = { C, Pm, S, -1.0, 1.0 };
    do_gemm(&gS, cfg, pool);

    /* 4) T = S^-1 (recursiva) -> O22 */
    if (inv_rec(S, O22, cfg, pool)) { arena_release(mk); return -1; }

    /* 5) M12 = -P*T  y  M21 = -T*Q   (independientes) */
    Gemm g12 = { Pm, O22, O12, -1.0, 0.0 }, g21 = { O22, Qm, O21, -1.0, 0.0 };
    do_pair(&g12, &g21, cfg, pool);

    /* 6) M11 = X - M12*Q   (P*T*Q = -M12*Q) */
    Gemm g11 = { O12, Qm, O11, -1.0, 1.0 };
    do_gemm(&g11, cfg, pool);

    arena_release(mk);
    return 0;
}

int invert(Mat M, Mat Out, const Cfg *cfg) {
    if (cfg->v == V_THREADS_BARRIER) {
        ThreadPool pool;
        pool_init(&pool, cfg->P);
        int rc = inv_rec(M, Out, cfg, &pool);
        pool_destroy(&pool);
        return rc;
    }
    return inv_rec(M, Out, cfg, NULL);
}
