#define _GNU_SOURCE
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>
#include <errno.h>
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

/* ================= Pool persistente de procesos ================= */
typedef struct {
    sem_t start;
    sem_t done;
    int stop;
    Gemm g;
    int r0, r1;
} ProcSlot;

static int       g_np    = 0;
static ProcSlot *g_slots = NULL;
static pid_t    *g_pids  = NULL;

static void sem_wait_intr(sem_t *s) {
    while (sem_wait(s) == -1) {
        if (errno != EINTR) { perror("sem_wait"); _exit(1); }
    }
}

static void proc_worker_loop(ProcSlot *slot) {
    for (;;) {
        sem_wait_intr(&slot->start);
        if (slot->stop) break;
        if (slot->r0 < slot->r1) gemm_rows(&slot->g, slot->r0, slot->r1);
        sem_post(&slot->done);
    }
}

void procs_pool_init(int P) {
    g_np = P < 1 ? 1 : P;
    size_t ndoubles = ((size_t)g_np * sizeof(ProcSlot) + sizeof(double) - 1) / sizeof(double);
    Mat shm_blk = arena_mat(1, (int)ndoubles);
    g_slots = (ProcSlot *)shm_blk.p;
    g_pids = malloc((size_t)g_np * sizeof(pid_t));
    if (!g_pids) { perror("malloc"); exit(1); }
    for (int i = 0; i < g_np; i++) {
        g_slots[i].stop = 0;
        if (sem_init(&g_slots[i].start, 1, 0) || sem_init(&g_slots[i].done, 1, 0)) {
            perror("sem_init"); exit(1);
        }
    }
    for (int i = 0; i < g_np; i++) {
        pid_t pid = fork();
        if (pid < 0) { perror("fork"); exit(1); }
        if (pid == 0) {
            proc_worker_loop(&g_slots[i]);
            _exit(0);
        }
        g_pids[i] = pid;
    }
}

static void procs_pool_gemm(const Gemm *g) {
    int rows = g->C.r;
    for (int i = 0; i < g_np; i++) {
        g_slots[i].stop = 0;
        g_slots[i].g = *g;
        g_slots[i].r0 = (int)((long)rows * i / g_np);
        g_slots[i].r1 = (int)((long)rows * (i + 1) / g_np);
        sem_post(&g_slots[i].start);
    }
    for (int i = 0; i < g_np; i++) sem_wait_intr(&g_slots[i].done);
}

void procs_pool_destroy(void) {
    if (!g_slots || !g_pids) return;
    for (int i = 0; i < g_np; i++) {
        g_slots[i].stop = 1;
        sem_post(&g_slots[i].start);
    }
    for (int i = 0; i < g_np; i++) {
        int st;
        while (waitpid(g_pids[i], &st, 0) == -1 && errno == EINTR) {}
        sem_destroy(&g_slots[i].start);
        sem_destroy(&g_slots[i].done);
    }
    free(g_pids);
    g_pids = NULL;
    g_slots = NULL;
    g_np = 0;
}

/* ================= Procesos con pipes (búfer local) ================= */
static void write_all(int fd, const void *buf, size_t n) {
    const char *p = buf;
    size_t off = 0;
    while (off < n) {
        ssize_t w = write(fd, p + off, n - off);
        if (w < 0) {
            if (errno == EINTR) continue;
            perror("write"); _exit(1);
        }
        off += (size_t)w;
    }
}

static void read_all(int fd, void *buf, size_t n) {
    char *p = buf;
    size_t off = 0;
    while (off < n) {
        ssize_t r = read(fd, p + off, n - off);
        if (r < 0) {
            if (errno == EINTR) continue;
            perror("read"); exit(1);
        }
        if (r == 0) { fprintf(stderr, "pipe EOF prematuro\n"); exit(1); }
        off += (size_t)r;
    }
}

static void procs_pipe_gemm(const Gemm *g, int np) {
    int rows = g->C.r, cols = g->C.c;
    if (np > rows) np = rows;
    if (np > 64) np = 64;
    if (np <= 1) { gemm_rows(g, 0, rows); return; }

    pid_t pids[64];
    int rfds[64], r0s[64], r1s[64];

    for (int p = 0; p < np; p++) {
        int pfd[2];
        if (pipe(pfd) < 0) { perror("pipe"); exit(1); }
        int r0 = (int)((long)rows * p / np);
        int r1 = (int)((long)rows * (p + 1) / np);
        r0s[p] = r0;
        r1s[p] = r1;

        pid_t pid = fork();
        if (pid < 0) { perror("fork"); exit(1); }
        if (pid == 0) {
            for (int k = 0; k < p; k++) close(rfds[k]);
            close(pfd[0]);
            int nr = r1 - r0;
            size_t bytes = (size_t)nr * cols * sizeof(double);
            double *buf = malloc(bytes ? bytes : 1);
            if (!buf) { perror("malloc"); _exit(1); }
            if (g->beta != 0.0) {
                for (int i = 0; i < nr; i++)
                    memcpy(buf + (size_t)i * cols, at(g->C, r0 + i, 0), (size_t)cols * sizeof(double));
            }
            Gemm lg = { sub(g->A, r0, 0, nr, g->A.c), g->B, { buf, nr, cols, cols }, g->alpha, g->beta };
            gemm_rows(&lg, 0, nr);
            write_all(pfd[1], buf, bytes);
            free(buf);
            close(pfd[1]);
            _exit(0);
        }
        close(pfd[1]);
        rfds[p] = pfd[0];
        pids[p] = pid;
    }

    for (int p = 0; p < np; p++) {
        int nr = r1s[p] - r0s[p];
        size_t bytes = (size_t)nr * cols * sizeof(double);
        double *buf = malloc(bytes ? bytes : 1);
        if (!buf) { perror("malloc"); exit(1); }
        read_all(rfds[p], buf, bytes);
        close(rfds[p]);
        for (int i = 0; i < nr; i++)
            memcpy(at(g->C, r0s[p] + i, 0), buf + (size_t)i * cols, (size_t)cols * sizeof(double));
        free(buf);
    }

    for (int p = 0; p < np; p++) {
        int st;
        while (waitpid(pids[p], &st, 0) == -1 && errno == EINTR) {}
        if (!WIFEXITED(st) || WEXITSTATUS(st)) { fprintf(stderr, "hijo fallo\n"); exit(1); }
    }
}

/* ================= Pool de hilos con cola dinámica (mutex + cond) ================= */
#define QUEUE_BLK 32

typedef struct {
    int P;
    int stop;
    const Gemm *cur_g;
    int next_row;
    int total_rows;
    int pending;
    int blocks_done;
    int total_blocks;
    pthread_mutex_t mtx;
    pthread_cond_t cond_work;
    pthread_cond_t cond_done;
    pthread_t *th;
} QueuePool;

static QueuePool *g_qpool = NULL;

static void *qpool_worker(void *arg) {
    QueuePool *q = arg;
    pthread_mutex_lock(&q->mtx);
    for (;;) {
        while (!q->stop && q->next_row >= q->total_rows)
            pthread_cond_wait(&q->cond_work, &q->mtx);
        if (q->stop) break;

        int r0 = q->next_row;
        int r1 = r0 + QUEUE_BLK;
        if (r1 > q->total_rows) r1 = q->total_rows;
        q->next_row = r1;
        const Gemm *g = q->cur_g;
        pthread_mutex_unlock(&q->mtx);

        gemm_rows(g, r0, r1);

        pthread_mutex_lock(&q->mtx);
        q->blocks_done++;
        if (--q->pending == 0)
            pthread_cond_signal(&q->cond_done);
    }
    pthread_mutex_unlock(&q->mtx);
    return NULL;
}

static void qpool_init(QueuePool *q, int P) {
    q->P = P < 1 ? 1 : P;
    q->stop = 0;
    q->cur_g = NULL;
    q->next_row = 0;
    q->total_rows = 0;
    q->pending = 0;
    q->blocks_done = 0;
    q->total_blocks = 0;
    q->th = NULL;
    if (q->P <= 1) return;
    q->th = malloc((size_t)q->P * sizeof(pthread_t));
    if (!q->th) { perror("malloc"); exit(1); }
    if (pthread_mutex_init(&q->mtx, NULL) ||
        pthread_cond_init(&q->cond_work, NULL) ||
        pthread_cond_init(&q->cond_done, NULL)) {
        perror("pthread_mutex/cond_init"); exit(1);
    }
    for (int t = 0; t < q->P; t++) {
        if (pthread_create(&q->th[t], NULL, qpool_worker, q)) {
            perror("pthread_create"); exit(1);
        }
    }
}

static void qpool_gemm(QueuePool *q, const Gemm *g) {
    int rows = g->C.r;
    if (rows <= 0) return;
    if (q->P <= 1) { gemm_rows(g, 0, rows); return; }
    int nblk = (rows + QUEUE_BLK - 1) / QUEUE_BLK;
    pthread_mutex_lock(&q->mtx);
    q->cur_g = g;
    q->next_row = 0;
    q->total_rows = rows;
    q->pending = nblk;
    q->blocks_done = 0;
    q->total_blocks = nblk;
    pthread_cond_broadcast(&q->cond_work);
    while (q->pending > 0)
        pthread_cond_wait(&q->cond_done, &q->mtx);
    if (q->blocks_done != q->total_blocks) {
        fprintf(stderr, "threads_queue: bloques %d != %d\n", q->blocks_done, q->total_blocks);
        exit(1);
    }
    pthread_mutex_unlock(&q->mtx);
}

static void qpool_destroy(QueuePool *q) {
    if (q->P <= 1) return;
    pthread_mutex_lock(&q->mtx);
    q->stop = 1;
    pthread_cond_broadcast(&q->cond_work);
    pthread_mutex_unlock(&q->mtx);
    for (int t = 0; t < q->P; t++) pthread_join(q->th[t], NULL);
    pthread_mutex_destroy(&q->mtx);
    pthread_cond_destroy(&q->cond_work);
    pthread_cond_destroy(&q->cond_done);
    free(q->th);
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
    case V_PROCS_POOL:      procs_pool_gemm(g); break;
    case V_PROCS_PIPE:      procs_pipe_gemm(g, c->P); break;
    case V_THREADS_QUEUE:   qpool_gemm(g_qpool, g); break;
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
    if (cfg->v == V_THREADS_QUEUE) {
        QueuePool qp;
        qpool_init(&qp, cfg->P);
        g_qpool = &qp;
        int rc = inv_rec(M, Out, cfg, NULL);
        qpool_destroy(&qp);
        g_qpool = NULL;
        return rc;
    }
    return inv_rec(M, Out, cfg, NULL);
}
