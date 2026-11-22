/* Bare-Metal ARM Cortex-M33 (Raspberry Pi Pico 2 / RP2350) & Host-Testable USB-CDC Firmware
 * for Threebits Pure-Integer 4-Bit Neural Implant Kernels:
 *   1. Packed int4 x int8 Causal GRU Decoder (`chip/int4_gru.c`, 66.35 kB total ROM+SRAM)
 *   2. Zero-Backprop Local On-Chip Plasticity (`chip/c99_local_plasticity.c`, 3.27 kB SRAM)
 *
 * Host verification build (macOS / Linux, zero warnings under -Werror):
 *   cc -O3 -std=c99 -Wall -Wextra -Werror -DHOST_TEST \
 *      chip/firmware_rp2350/main.c chip/int4_gru.c chip/c99_local_plasticity.c \
 *      -o chip/firmware_rp2350/rp2350_host_emu
 *
 *   # Deterministic benchmark mode:
 *   ./chip/firmware_rp2350/rp2350_host_emu --bench
 *
 *   # Live USB-CDC / POSIX PTY wire-protocol mode (connects to level3_app/server.py --serial):
 *   ./chip/firmware_rp2350/rp2350_host_emu --stdio
 *
 * Raspberry Pi Pico 2 (RP2350, Cortex-M33 @ 150 MHz) build with Pico SDK:
 *   mkdir build && cd build && cmake -DPICO_BOARD=pico2 .. && make -j
 *   Flash `threebits_int4_gru.uf2` over USB, then run:
 *   .venv/bin/python level3_app/server.py --serial /dev/cu.usbmodem*
 */
#ifdef HOST_TEST
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef HOST_TEST
#include <time.h>
static uint64_t now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}
#define TB_BOARD_NAME "host_emu"
#else
#include "pico/stdlib.h"
static uint64_t now_us(void) { return time_us_64(); }
#define TB_BOARD_NAME "rp2350"
#endif

#include "falcon_h1_weights.h"
#include "rp2350_protocol.h"

/* Declarations from chip/int4_gru.c */
typedef struct {
    int32_t input_size, hidden_size, output_size;
    const uint8_t *w_ih_packed, *w_hh_packed, *w_out_packed;
    const int32_t *b_ih_q12, *b_hh_q12, *b_out_q12;
    int32_t ih_mult, ih_shift, hh_mult, hh_shift, out_mult, out_shift;
    const int16_t *sigmoid_lut, *tanh_lut;
} Int4GRUModel;

void gru_int4_step(const Int4GRUModel *m, const int8_t *x_q8, int8_t *h_q7,
                   int32_t *out_q12, int32_t *scratch_6h);

/* Declarations from chip/c99_local_plasticity.c */
typedef struct {
    int32_t n_in, n_codes, d_lat, lr_shift, v_ema_q8, n_updates;
    int16_t *w_read_q8;
    int16_t *w_write_q8;
    int8_t  *last_x_q8;
    int32_t *usage_q8;
    int32_t  last_pred, last_margin;
} C99LocalChip;

int32_t c99_local_forward(C99LocalChip *chip, const int8_t *x_q8,
                          int8_t *out_wb_q8, int32_t *out_scores_q8);
void c99_local_learn_rpe(C99LocalChip *chip, int32_t reward_bit, int32_t runner_up);

/* Strictly static SRAM buffers (zero malloc, deterministic footprint) */
static int8_t  g_hidden[TB_HIDDEN_SIZE];
static int32_t g_scratch[6 * TB_HIDDEN_SIZE];
static int32_t g_output[TB_OUTPUT_SIZE];
static int8_t  g_input[TB_INPUT_SIZE];

static int16_t g_w_read[TB_LOCAL_CODES * TB_LOCAL_IN];
static int16_t g_w_write[TB_LOCAL_CODES * TB_LOCAL_DLAT];
static int8_t  g_last_x[TB_LOCAL_IN];
static int32_t g_usage[TB_LOCAL_CODES];
static int8_t  g_wb_q8[TB_LOCAL_DLAT];
static int32_t g_local_scores[TB_LOCAL_CODES];
static C99LocalChip g_local_chip;

static char g_line_buf[TB_LINE_MAX];
static char g_reply_buf[TB_LINE_MAX];
static uint32_t g_gru_seq = 0;
static uint32_t g_local_seq = 0;

static const Int4GRUModel g_gru_model = {
    TB_INPUT_SIZE, TB_HIDDEN_SIZE, TB_OUTPUT_SIZE,
    TB_W_IH_PACKED, TB_W_HH_PACKED, TB_W_OUT_PACKED,
    TB_B_IH_Q12, TB_B_HH_Q12, TB_B_OUT_Q12,
    TB_IH_MULT, TB_IH_SHIFT, TB_HH_MULT, TB_HH_SHIFT, TB_OUT_MULT, TB_OUT_SHIFT,
    TB_SIGMOID_LUT, TB_TANH_LUT
};

static size_t gru_rom_bytes(void) {
    return sizeof(TB_W_IH_PACKED) + sizeof(TB_W_HH_PACKED) + sizeof(TB_W_OUT_PACKED)
         + sizeof(TB_B_IH_Q12) + sizeof(TB_B_HH_Q12) + sizeof(TB_B_OUT_Q12)
         + sizeof(TB_SIGMOID_LUT) + sizeof(TB_TANH_LUT);
}

static size_t total_sram_bytes(void) {
    return sizeof(g_hidden) + sizeof(g_scratch) + sizeof(g_output) + sizeof(g_input)
         + sizeof(g_w_read) + sizeof(g_w_write) + sizeof(g_last_x) + sizeof(g_usage)
         + sizeof(g_wb_q8) + sizeof(g_local_scores) + sizeof(g_local_chip);
}

static void reset_all_state(void) {
    for (int j = 0; j < TB_HIDDEN_SIZE; ++j) g_hidden[j] = 0;
    for (int i = 0; i < TB_INPUT_SIZE; ++i) g_input[i] = 0;

    uint32_t rng = 0x13579BDFu;
    for (int i = 0; i < TB_LOCAL_CODES * TB_LOCAL_IN; ++i) {
        rng = rng * 1664525u + 1013904223u;
        g_w_read[i] = (int16_t)((int32_t)((rng >> 24) & 0x5Fu) - 48);
    }
    for (int c = 0; c < TB_LOCAL_CODES; ++c) {
        g_usage[c] = (256 * 16) / TB_LOCAL_CODES;
        for (int d = 0; d < TB_LOCAL_DLAT; ++d) g_w_write[c * TB_LOCAL_DLAT + d] = 0;
        g_w_write[c * TB_LOCAL_DLAT + (2 * c)] = 224;
        g_w_write[c * TB_LOCAL_DLAT + (2 * c + 1)] = 224;
    }
    g_local_chip.n_in = TB_LOCAL_IN;
    g_local_chip.n_codes = TB_LOCAL_CODES;
    g_local_chip.d_lat = TB_LOCAL_DLAT;
    g_local_chip.lr_shift = 4;
    g_local_chip.v_ema_q8 = 32;
    g_local_chip.n_updates = 0;
    g_local_chip.w_read_q8 = g_w_read;
    g_local_chip.w_write_q8 = g_w_write;
    g_local_chip.last_x_q8 = g_last_x;
    g_local_chip.usage_q8 = g_usage;
    g_local_chip.last_pred = 0;
    g_local_chip.last_margin = 0;
    g_gru_seq = 0;
    g_local_seq = 0;
}

static void fill_seeded_input(uint32_t seed, int8_t *dst, int n) {
    uint32_t s = seed ^ 0x9E3779B9u;
    for (int i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        dst[i] = (int8_t)((int32_t)((s >> 24) % 61u) - 30);
    }
}

static void emit_crc_line(const char *payload) {
    size_t n = strlen(payload);
    uint16_t crc = tb_crc16_ccitt((const uint8_t *)payload, n);
    printf("%s crc=%04x\n", payload, (unsigned)crc);
    fflush(stdout);
}

static void run_deterministic_bench(int n_steps, int machine_fmt) {
    reset_all_state();
    for (int i = 0; i < TB_INPUT_SIZE; ++i) g_input[i] = (int8_t)((i * 37) % 61 - 30);

    uint64_t t0 = now_us();
    int64_t checksum = 0;
    for (int s = 0; s < n_steps; ++s) {
        g_input[s % TB_INPUT_SIZE] = (int8_t)((s * 13) % 53 - 26);
        gru_int4_step(&g_gru_model, g_input, g_hidden, g_output, g_scratch);
        checksum += g_output[0];
    }
    uint64_t gru_dt_us = now_us() - t0;

    uint64_t t1 = now_us();
    int64_t local_chk = 0;
    for (int s = 0; s < n_steps; ++s) {
        int tok = s & 7;
        for (int i = 0; i < TB_LOCAL_IN; ++i) g_input[i] = 0;
        for (int k = 0; k < 16; ++k) g_input[(tok * 16 + k) % 140] = 64;
        int32_t pred = c99_local_forward(&g_local_chip, g_input, g_wb_q8, g_local_scores);
        int32_t runner = (pred == 0) ? 1 : 0;
        for (int c = 0; c < TB_LOCAL_CODES; ++c) {
            if (c != pred && g_local_scores[c] > g_local_scores[runner]) runner = c;
        }
        c99_local_learn_rpe(&g_local_chip, (s & 1), runner);
        local_chk += pred + g_wb_q8[0];
    }
    uint64_t local_dt_us = now_us() - t1;

    if (machine_fmt) {
        snprintf(g_reply_buf, sizeof(g_reply_buf),
                 "BENCH_OUT steps=%d gru_total_us=%llu local_total_us=%llu gru_chk=%lld local_chk=%lld",
                 n_steps, (unsigned long long)gru_dt_us, (unsigned long long)local_dt_us,
                 (long long)checksum, (long long)local_chk);
        emit_crc_line(g_reply_buf);
        return;
    }

    uint64_t gru_ns_per_step = (n_steps > 0) ? ((gru_dt_us * 1000ULL) / (uint64_t)n_steps) : 0ULL;
    uint64_t local_ns_per_step = (n_steps > 0) ? ((local_dt_us * 1000ULL) / (uint64_t)n_steps) : 0ULL;
    printf("=== Threebits Pure-Integer 4-Bit Human Implant Firmware (%s) ===\n", TB_BOARD_NAME);
    printf("Config:   GRU %d->%d->%d + Local %d->%d->%d | Flash ROM: %zu B | Static SRAM: %zu B\n",
           TB_INPUT_SIZE, TB_HIDDEN_SIZE, TB_OUTPUT_SIZE,
           TB_LOCAL_IN, TB_LOCAL_CODES, TB_LOCAL_DLAT,
           gru_rom_bytes(), total_sram_bytes());
    printf("GRU:      %d steps in %llu us -> %llu.%03llu us / 20 ms bin | checksum=%lld | out[0..2]=[%d,%d,%d]\n",
           n_steps, (unsigned long long)gru_dt_us,
           (unsigned long long)(gru_ns_per_step / 1000ULL),
           (unsigned long long)(gru_ns_per_step % 1000ULL),
           (long long)checksum, g_output[0], g_output[1], g_output[2]);
    printf("Local:    %d trials in %llu us -> %llu.%03llu us / trial | updates=%d | checksum=%lld\n",
           n_steps, (unsigned long long)local_dt_us,
           (unsigned long long)(local_ns_per_step / 1000ULL),
           (unsigned long long)(local_ns_per_step % 1000ULL),
           g_local_chip.n_updates, (long long)local_chk);
    fflush(stdout);
}

static void handle_command_line(char *line) {
    /* Strip trailing CR/LF */
    size_t len = strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r' || line[len - 1] == ' ')) {
        line[--len] = '\0';
    }
    if (len == 0) return;

    if (strcmp(line, "PING") == 0) {
        snprintf(g_reply_buf, sizeof(g_reply_buf),
                 "PONG proto=%d board=%s rom_b=%zu sram_b=%zu gru=%d,%d,%d local=%d,%d,%d",
                 TB_PROTO_VERSION, TB_BOARD_NAME, gru_rom_bytes(), total_sram_bytes(),
                 TB_INPUT_SIZE, TB_HIDDEN_SIZE, TB_OUTPUT_SIZE,
                 TB_LOCAL_IN, TB_LOCAL_CODES, TB_LOCAL_DLAT);
        emit_crc_line(g_reply_buf);
        return;
    }

    if (strcmp(line, "RESET") == 0) {
        reset_all_state();
        emit_crc_line("OK RESET");
        return;
    }

    if (strncmp(line, "BENCH", 5) == 0) {
        int steps = 200;
        if (line[5] == ' ') {
            long v = strtol(line + 6, NULL, 10);
            if (v >= 1 && v <= 20000) steps = (int)v;
        }
        run_deterministic_bench(steps, 1);
        return;
    }

    if (strncmp(line, "GRU ", 4) == 0) {
        const char *arg = line + 4;
        if (strncmp(arg, "SEED ", 5) == 0) {
            uint32_t seed = (uint32_t)strtoul(arg + 5, NULL, 10);
            fill_seeded_input(seed, g_input, TB_INPUT_SIZE);
        } else {
            if (strlen(arg) < (size_t)(2 * TB_INPUT_SIZE) ||
                tb_hex_to_i8(arg, g_input, TB_INPUT_SIZE) != 0) {
                emit_crc_line("ERR BAD_GRU_HEX");
                return;
            }
        }
        uint64_t t0 = now_us();
        gru_int4_step(&g_gru_model, g_input, g_hidden, g_output, g_scratch);
        uint64_t dt = now_us() - t0;
        g_gru_seq += 1;
        snprintf(g_reply_buf, sizeof(g_reply_buf),
                 "GRU_OUT seq=%lu us=%llu out=%d,%d,%d,%d,%d,%d,%d",
                 (unsigned long)g_gru_seq, (unsigned long long)dt,
                 (int)g_output[0], (int)g_output[1], (int)g_output[2], (int)g_output[3],
                 (int)g_output[4], (int)g_output[5], (int)g_output[6]);
        emit_crc_line(g_reply_buf);
        return;
    }

    if (strncmp(line, "LOCAL ", 6) == 0) {
        char *p = line + 6;
        int32_t reward_bit = (int32_t)strtol(p, &p, 10);
        while (*p == ' ') ++p;
        if (strncmp(p, "SEED ", 5) == 0) {
            uint32_t seed = (uint32_t)strtoul(p + 5, NULL, 10);
            fill_seeded_input(seed, g_input, TB_LOCAL_IN);
        } else {
            if (strlen(p) < (size_t)(2 * TB_LOCAL_IN) ||
                tb_hex_to_i8(p, g_input, TB_LOCAL_IN) != 0) {
                emit_crc_line("ERR BAD_LOCAL_HEX");
                return;
            }
        }
        uint64_t t0 = now_us();
        int32_t pred = c99_local_forward(&g_local_chip, g_input, g_wb_q8, g_local_scores);
        if (reward_bit >= 0) {
            int32_t runner = (pred == 0) ? 1 : 0;
            for (int c = 0; c < TB_LOCAL_CODES; ++c) {
                if (c != pred && g_local_scores[c] > g_local_scores[runner]) runner = c;
            }
            c99_local_learn_rpe(&g_local_chip, reward_bit ? 1 : 0, runner);
        }
        uint64_t dt = now_us() - t0;
        g_local_seq += 1;
        char wb_hex[2 * TB_LOCAL_DLAT + 1];
        tb_i8_to_hex(g_wb_q8, TB_LOCAL_DLAT, wb_hex);
        snprintf(g_reply_buf, sizeof(g_reply_buf),
                 "LOCAL_OUT seq=%lu us=%llu pred=%d margin=%d updates=%d wb=%s",
                 (unsigned long)g_local_seq, (unsigned long long)dt,
                 (int)pred, (int)g_local_chip.last_margin, (int)g_local_chip.n_updates, wb_hex);
        emit_crc_line(g_reply_buf);
        return;
    }

    emit_crc_line("ERR UNKNOWN_CMD");
}

static void run_stdio_protocol_loop(void) {
    reset_all_state();
    setvbuf(stdout, NULL, _IONBF, 0);
    while (fgets(g_line_buf, (int)sizeof(g_line_buf), stdin) != NULL) {
        handle_command_line(g_line_buf);
    }
}

int main(int argc, char **argv) {
#ifndef HOST_TEST
    stdio_init_all();
    sleep_ms(1200);
    (void)argc;
    (void)argv;
    /* Announce boot readiness on USB-CDC, then serve real-time host commands continuously */
    reset_all_state();
    snprintf(g_reply_buf, sizeof(g_reply_buf),
             "READY proto=%d board=%s rom_b=%zu sram_b=%zu",
             TB_PROTO_VERSION, TB_BOARD_NAME, gru_rom_bytes(), total_sram_bytes());
    emit_crc_line(g_reply_buf);
    run_stdio_protocol_loop();
#else
    if (argc > 1 && strcmp(argv[1], "--stdio") == 0) {
        run_stdio_protocol_loop();
        return 0;
    }
    int steps = 2000;
    if (argc > 2 && strcmp(argv[1], "--bench") == 0) {
        long v = strtol(argv[2], NULL, 10);
        if (v >= 1 && v <= 20000) steps = (int)v;
    }
    run_deterministic_bench(steps, 0);
#endif
    return 0;
}
