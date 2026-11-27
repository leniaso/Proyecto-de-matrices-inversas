# Inversión de matrices con procesos e hilos jerarquizados

Proyecto final del curso de **Sistemas Operativos y Laboratorio**
Universidad de Antioquia · Facultad de Ingeniería

**Equipo:** Alejandro Naranjo Cataño y Simon Correa Rios

Calcula la inversa de matrices grandes con un esquema de divide y vencerás
(complemento de Schur) en C sobre Linux, y mide con estadística si repartir el
trabajo con **procesos e hilos** reduce el tiempo frente a una versión
secuencial, una de solo hilos y una de solo procesos, además de comparar distintos
mecanismos de sincronización y comunicación (pools persistentes, barreras,
semáforos POSIX, colas dinámicas y pipes).

## Requisitos

- Linux o WSL (el código usa `fork`, `shm_open`, `mmap`, `pthread` y semáforos POSIX; no compila en Windows nativo).
- `gcc` y `make` (en Ubuntu: `sudo apt install build-essential`).
- Para el análisis: Python 3 con `pandas`, `numpy`, `scipy`, `statsmodels` y `matplotlib`.

## Compilar y probar

```bash
make
./matinv --header
for v in seq threads procs hier threads_barrier procs_pool procs_pipe threads_queue; do
  ./matinv $v 500 4 32 7
done
```

Todas las líneas deben terminar en `ok` y con un residuo cercano a `1e-14`.

## Uso

```
./matinv <versión> <n> <P> <n0> <semilla> [singular]
./matinv --header
```

| Parámetro | Significado |
|---|---|
| `versión` | `seq`, `threads`, `procs`, `hier`, `pairt`, `threads_barrier`, `procs_pool`, `procs_pipe` o `threads_queue` |
| `n` | Tamaño de la matriz (n × n). No tiene que ser potencia de 2 |
| `P` | Total de unidades de cómputo (hilos, procesos, o procesos × hilos). Se ignora en `seq` |
| `n0` | Tamaño de corte: los bloques de tamaño menor o igual se resuelven con Gauss-Jordan |
| `semilla` | Semilla del generador de la matriz (diagonal dominante) |
| `singular` | Opcional: repite una fila para probar la detección de matrices singulares |

Cada ejecución imprime una línea CSV:

```
version,n,P,n0,seed,time_s,residual,status,maxrss_kb,vol_ctx,invol_ctx
```

- `time_s`: tiempo de pared de la inversión (`clock_gettime(CLOCK_MONOTONIC)`); el residuo se calcula fuera del cronómetro.
- `residual`: ‖A·A⁻¹ − I‖∞. Vale `-1` si la matriz es singular y `-2` si no se calculó (`NORES`).
- `maxrss_kb`, `vol_ctx`, `invol_ctx`: memoria máxima y cambios de contexto voluntarios e involuntarios.

### Variables de entorno

| Variable | Efecto |
|---|---|
| `NORES=1` | No calcula el residuo (ahorra mucho tiempo en n grande) |
| `DUMP=prefijo` | Guarda la matriz y su inversa en `prefijo_M.bin` y `prefijo_Inv.bin` para validar con NumPy |
| `ARENA_ANON=1` | Usa memoria compartida anónima en lugar de `shm_open` (si `/dev/shm` es muy pequeño, por ejemplo en Docker) |

## Versiones

Cada nivel de la recursión hace 2 inversiones recursivas y 6 productos de bloques;
los pares (P y Q) y (M₁₂ y M₂₁) son independientes. Las versiones se diferencian en
cómo ejecutan y sincronizan los productos:

| Versión | Estrategia |
|---|---|
| `seq` | Todo secuencial (línea base) |
| `threads` | Cada producto se reparte por filas entre P hilos creados y unidos por llamada (`pthread_create`/`pthread_join`) |
| `procs` | Cada producto se reparte por filas entre P procesos (`fork`/`waitpid`), con los datos en memoria compartida (`shm_open` + `mmap`) |
| `hier` | Los pares independientes corren en dos procesos y, dentro de cada uno, P/2 hilos reparten las filas; los productos dependientes usan los P hilos |
| `pairt` | Diagnóstico: igual que `hier` pero con hilos en lugar de procesos. No forma parte del diseño factorial |
| `threads_barrier` | Pool persistente de P hilos sincronizados con barreras (`pthread_barrier_t`) y reparto estático por filas |
| `procs_pool` | Pool persistente de P procesos hijos sincronizados con semáforos POSIX en memoria compartida (`sem_t`) |
| `procs_pipe` | P procesos hijos por producto que devuelven su bloque de filas al padre mediante `pipe()` con búfer local |
| `threads_queue` | Pool persistente de P hilos con cola dinámica de bloques de 32 filas sincronizados con `pthread_mutex_t` y `pthread_cond_t` |

Los hilos y procesos escriben filas distintas del resultado y solo leen A y B, así que
no hay condiciones de carrera ni se necesitan mutex dentro del producto.

## Scripts

| Archivo | Para qué sirve |
|---|---|
| `bench.sh` | Diseño factorial base (`seq`, `threads`, `procs`, `hier`) con réplicas, calentamiento y orden aleatorizado. Calcula el residuo solo en la réplica 1 |
| `bench_pool.sh` | Diseño factorial extendido para comparar las variantes de sincronización y comunicación (`VERS` configurable) |
| `validate.py` | Compara la inversa con `numpy.linalg.inv` (LAPACK) |
| `analisis.py` | Análisis estadístico del experimento base (`results.csv`): tablas, figuras y `resumen.txt` |
| `analisis_pool.py` | Análisis estadístico de `results_pool.csv` con todas las variantes paralelas |
| `analisis_comparaciones.py` | Pruebas por pares (Mann-Whitney con corrección de Holm) para comparar pool vs creación por producto, memoria compartida vs pipes y cola dinámica vs barreras |

### Validar contra LAPACK

```bash
DUMP=/tmp/t ./matinv hier 777 4 40 9
python3 validate.py /tmp/t 777
```

### Reproducir los experimentos

Experimento base:

```bash
chmod +x bench.sh
NC=$(( $(nproc) - 1 ))
SIZES="512 1024 2048 4096" REPS=30 REPS_BIG=10 N0=128 \
TASKSET="taskset -c 0-$NC" OUT=results.csv ./bench.sh
```

Experimento de variantes de sincronización y comunicación (`pool`, `pipe`, `queue`):

```bash
chmod +x bench_pool.sh
NC=$(( $(nproc) - 1 ))
SIZES="512 1024 2048 4096" REPS=10 REPS_BIG=8 N0=128 \
TASKSET="taskset -c 0-$NC" OUT=results_pool.csv ./bench_pool.sh
```

### Analizar

```bash
python3 -m venv venv && source venv/bin/activate
pip install pandas numpy scipy statsmodels matplotlib
python3 analisis.py results.csv resultados_kernel2
python3 analisis_pool.py results_pool.csv resultados_pool
python3 analisis_comparaciones.py results_pool.csv resultados_comparaciones
```

## Datos y resultados

| Archivo o carpeta | Contenido |
|---|---|
| `results.csv` | Experimento principal definitivo (kernel con bloqueo de 4 filas) |
| `results_kernel1.csv` | Experimento principal con el producto de matrices original (etiqueta Git `kernel-v1`) |
| `results_pool.csv` | Experimento con las 7 versiones paralelas (`threads`, `procs`, `hier`, `threads_barrier`, `procs_pool`, `procs_pipe`, `threads_queue`) y `seq` |
| `resultados_kernel1/`, `resultados_kernel2/` | Tablas, figuras y `resumen.txt` de cada experimento base |
| `resultados_kernel1_primera_corrida/` | Primera corrida de referencia con el kernel original |
| `resultados_pool/` | Tablas, figuras y `resumen.txt` del análisis de `results_pool.csv` |
| `resultados_comparaciones/` | `comparaciones_pares.csv` con las pruebas por pares entre mecanismos |
| `piloto_*.csv`, `piloto15_*.csv` | Piloto para elegir n0 (n = 1024) |
| `cache_test.csv`, `pairt_test.csv`, `pin_test.csv`, `tiling_test.csv` | Pruebas de diagnóstico de la anomalía en n grande |
| `datos_respaldo/` | Respaldo de CSVs de corridas anteriores |
| `prueba_pool.csv`, `prueba_pool_out/`, `prueba_tiempo.csv` | Pruebas preliminares del pool |

### Resumen de resultados (experimento base)

Medidos en un AMD Ryzen 5 5600 (6 núcleos físicos, 12 hilos), con `gcc -O2`, n0 = 128.

- La versión jerárquica es significativamente más rápida que la secuencial en las
  12 combinaciones de n y P, con reducciones de tiempo de 29 % a 74 %.
- Frente a solo procesos gana en 6 de 12 combinaciones (hasta 27 % más rápida), pero pierde en n = 4096.
- Frente a solo hilos gana en 2 de 12 (por 2 %) y pierde significativamente en 8 de 12;
  en n = 4096 es entre 11 % y 28 % más lenta. No se puede afirmar que la jerárquica supere a solo hilos.
- Mejor aceleración en el experimento base: `threads` con P = 8 y n = 4096, de 4,41× (20,2 s a 4,58 s).
- Todos los residuos verificados son menores que 3e-14 (umbral de aceptación: 1e-8).

La primera versión del producto de matrices dejaba el programa limitado por la
memoria en n grande y exageraba la desventaja de `hier` (hasta 74 % más lenta que
`threads` en n = 4096). Se cambió el producto para leer menos veces la matriz B y
se repitió todo el experimento. La causa es consistente con el ancho de banda de
memoria, pero no está demostrada con una medición directa.

### Resumen de resultados (variantes de sincronización y comunicación)

Sobre 836 corridas en `results_pool.csv` (88 con residuo verificado, máximo 2,929e-14):

- Mejor aceleración global: `threads_queue` con P = 8 y n = 4096, de **4,70×** (19,41 s a 4,13 s), seguido de `procs_pool` (4,57×) y `threads_barrier` (4,54×).
- **Pool de procesos vs `fork` por producto (`procs_pool` vs `procs`):** `procs_pool` es significativamente más rápido en 10 de 12 combinaciones (reducciones de 5 % a 31,6 %).
- **Memoria compartida vs pipes (`procs` vs `procs_pipe`):** la memoria compartida supera significativamente a los pipes en las 12 combinaciones (reducciones de 8,1 % a 20,3 %).
- **Pool con barreras vs `pthread_create` por producto (`threads_barrier` vs `threads`):** `threads_barrier` gana significativamente en 5 de 12 combinaciones (principalmente n = 512 y n = 1024 con P = 4 y 8, entre 5,1 % y 11,0 %).
- **Cola dinámica vs barreras (`threads_queue` vs `threads_barrier`):** `threads_barrier` es 4,1 % más rápida en n = 512, P = 2, mientras que `threads_queue` es significativamente más rápida en P = 8 para n = 1024 (−12,4 %) y n = 2048 (−6,4 %).

## Estructura del código

| Archivo | Contenido |
|---|---|
| `main.c` | Lee los parámetros, genera la matriz, cronometra la inversión y escribe la línea CSV |
| `invert.c` | Recursión por bloques con complemento de Schur, pares independientes y pools/pipes/colas (`threads_barrier`, `procs_pool`, `procs_pipe`, `threads_queue`) |
| `matrix.c` / `matrix.h` | Vistas de matriz, arena en memoria compartida, Gauss-Jordan, residuo y producto de matrices por filas (hilos o procesos) |
| `Makefile` | Compila con `gcc -O2 -Wall -Wextra -pthread` |

## Pendiente

- Pruebas de corrección sistemáticas (inversas conocidas, más tamaños y casos singulares).
- Repetir el piloto de n0 con el kernel actual.
- Informe final.

## Notas

- Si `pip` responde `externally-managed-environment`, usar el entorno virtual indicado arriba.
- No se versionan el binario `matinv` ni la carpeta `venv/` (están en `.gitignore`).
- Si no se puede fijar el gobernador de CPU (por ejemplo en WSL), documentarlo junto con los resultados.
