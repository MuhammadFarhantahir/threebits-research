/* Standalone Bare-Metal Pure-Integer C99 On-Chip Local Co-Adaptation Benchmark (RP2350 / Cortex-M33).
 *
 * Compiles with ZERO floating-point math inside the forward/learning loop and ZERO malloc().
 * Uses static SRAM buffers only (~3.23 kB total static footprint for 176 ch -> 8 codes -> 16-d writeback).
 *
 * Host build & run:
 *     cc -O3 -Wall -Wextra chip/firmware_rp2350/coadapt_bench.c chip/c99_local_plasticity.c -o chip/firmware_rp2350/coadapt_bench
 *     ./chip/firmware_rp2350/coadapt_bench
 */
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define N_IN 176
#define N_CODES 8
#define D_LAT 16

typedef struct {
    int32_t n_in, n_codes, d_lat, lr_shift, v_ema_q8, n_updates;
    int16_t *w_read_q8;
    int16_t *w_write_q8;
    int8_t  *last_x_q8;
    int32_t *usage_q8;
    int32_t  last_pred, last_margin;
} C99LocalChip;

int32_t c99_local_forward(C99LocalChip *chip, const int8_t *x_q8, int8_t *out_wb_q8, int32_t *out_scores_q8);
void c99_local_learn_rpe(C99LocalChip *chip, int32_t reward_bit, int32_t runner_up);

/* Strictly static SRAM arrays (zero dynamic heap allocation) */
static int16_t g_w_read[N_CODES * N_IN];
static int16_t g_w_write[N_CODES * D_LAT];
static int8_t  g_last_x[N_IN];
static int32_t g_usage[N_CODES];

int main(void) {
    uint32_t rng = 0x13579BDFu;
    for (int i = 0; i < N_CODES * N_IN; ++i) {
        rng = rng * 1664525u + 1013904223u;
        g_w_read[i] = (int16_t)((int32_t)((rng >> 24) & 0x5Fu) - 48);
    }
    for (int c = 0; c < N_CODES; ++c) {
        g_usage[c] = (256 * 16) / N_CODES;
        for (int d = 0; d < D_LAT; ++d) g_w_write[c * D_LAT + d] = 0;
        g_w_write[c * D_LAT + (2 * c)] = 224;
        g_w_write[c * D_LAT + (2 * c + 1)] = 224;
    }

    C99LocalChip chip = {
        N_IN, N_CODES, D_LAT, 4, 32, 0,
        g_w_read, g_w_write, g_last_x, g_usage, 0, 0
    };

    int8_t x_q8[N_IN];
    int8_t wb_q8[D_LAT];
    int32_t scores[N_CODES];
    const int N_ITERS = 20000;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int32_t checksum = 0;
    for (int it = 0; it < N_ITERS; ++it) {
        int tok = it & 7;
        for (int i = 0; i < N_IN; ++i) x_q8[i] = 0;
        for (int k = 0; k < 16; ++k) x_q8[(tok * 16 + k) % 140] = 64;

        int32_t pred = c99_local_forward(&chip, x_q8, wb_q8, scores);
        int32_t runner = (pred == 0) ? 1 : 0;
        for (int c = 0; c < N_CODES; ++c) {
            if (c != pred && scores[c] > scores[runner]) runner = c;
        }
        c99_local_learn_rpe(&chip, (it & 1), runner);
        checksum += pred + wb_q8[0];
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed_us = ((double)(t1.tv_sec - t0.tv_sec) * 1e6) + ((double)(t1.tv_nsec - t0.tv_nsec) * 1e-3);
    size_t static_bytes = sizeof(g_w_read) + sizeof(g_w_write) + sizeof(g_last_x) + sizeof(g_usage) + sizeof(C99LocalChip);
    printf("Threebits Zero-Backprop C99 On-Chip Local Learner (RP2350 / Cortex-M33 Bare-Metal)\n");
    printf("  Static SRAM/ROM footprint: %zu bytes (%.2f kB, 0 malloc, 0 floats)\n", static_bytes, static_bytes / 1024.0);
    printf("  Iterations: %d | Mean latency (forward + online STDP/Hebbian learn): %.3f us/trial\n", N_ITERS, elapsed_us / N_ITERS);
    printf("  Updates applied: %d | Checksum: %d\n", chip.n_updates, checksum);
    return 0;
}
