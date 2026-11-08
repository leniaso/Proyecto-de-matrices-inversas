#define _GNU_SOURCE
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "--header")) {
        puts("version,n,P,n0,seed,time_s,residual,status,maxrss_kb,vol_ctx,invol_ctx");
        return 0;
    }
    if (argc < 6) {
        fprintf(stderr, "uso: %s <seq|threads|procs|hier> <n> <P> <n0> <seed> [singular]\n"
                        "     %s --header\n"
                        "     DUMP=prefijo guarda matriz e inversa (binario) para validar con numpy\n",
                argv[0], argv[0]);
        return 1;
    }
    Cfg cfg; const char *vn = argv[1];
    if      (!strcmp(vn, "seq"))     cfg.v = V_SEQ;
    else if (!strcmp(vn, "threads")) cfg.v = V_THREADS;
    else if (!strcmp(vn, "procs"))   cfg.v = V_PROCS;
    else if (!strcmp(vn, "hier"))    cfg.v = V_HIER;
    else { fprintf(stderr, "version desconocida\n"); return 1; }
    int n = atoi(argv[2]); cfg.P = atoi(argv[3]); cfg.n0 = atoi(argv[4]);
    unsigned seed = (unsigned)atoi(argv[5]);
    if (cfg.v == V_SEQ) cfg.P = 1;
    if (cfg.P < 1) cfg.P = 1;

    size_t bytes = (size_t)6 * n * n * sizeof(double) + (1 << 20);
    if (arena_init(bytes)) return 1;

    Mat M = arena_mat(n, n), Out = arena_mat(n, n);
    mat_diagdom(M, seed);
    if (argc > 6 && !strcmp(argv[6], "singular"))
        memcpy(at(M, 1, 0), at(M, 0, 0), (size_t)n * sizeof(double));   /* fila repetida */

    double t0 = now();
    int rc = invert(M, Out, &cfg);
    double t1 = now();

    double res = rc == 0 ? residual_inf(M, Out) : -1.0;   /* fuera del cronometro */

    const char *dump = getenv("DUMP");
    if (dump && rc == 0) {
        char f[256]; FILE *fp;
        snprintf(f, sizeof f, "%s_M.bin", dump);   fp = fopen(f, "wb"); fwrite(M.p, sizeof(double), (size_t)n * n, fp); fclose(fp);
        snprintf(f, sizeof f, "%s_Inv.bin", dump); fp = fopen(f, "wb"); fwrite(Out.p, sizeof(double), (size_t)n * n, fp); fclose(fp);
    }

    struct rusage s, c; getrusage(RUSAGE_SELF, &s); getrusage(RUSAGE_CHILDREN, &c);
    long rss = s.ru_maxrss > c.ru_maxrss ? s.ru_maxrss : c.ru_maxrss;
    printf("%s,%d,%d,%d,%u,%.6f,%.3e,%s,%ld,%ld,%ld\n", vn, n, cfg.P, cfg.n0, seed,
           t1 - t0, res, rc == 0 ? "ok" : "singular", rss,
           s.ru_nvcsw + c.ru_nvcsw, s.ru_nivcsw + c.ru_nivcsw);
    arena_destroy();
    return rc == 0 ? 0 : 3;
}
