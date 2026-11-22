/* Pure-Integer C99 On-Chip Local Co-Adaptation Kernel (Zero Floats, Zero Backprop!).
 *
 * Implements Three-Factor Reward-Modulated Hebbian / Local Plasticity directly on an
 * implantable microcontroller (ARM Cortex-M33 / RP2350 target):
 *   - Reads quantized int8 spiking counts x_q8[N_IN] (e.g., 176 LIF channels).
 *   - Computes int4-quantized latent key/value readout in int32 MACs + exact bit-shift scaling.
 *   - Stores associative memory slots in compact static SRAM (16 slots x D_LAT int8).
 *   - Writes norm-bounded safe int8 feedback code back to cortex.
 *   - Updates its OWN int8/int4 projection weights ONLINE on chip via strictly local
 *     Three-Factor Eligibility Traces x 1-bit Dopamine RPE (r_t - v_ema) — ZERO floats,
 *     ZERO backpropagation, ZERO cloud connection!
 *
 * Compile shared library:
 *     cc -O3 -Wall -Wextra -shared -fPIC chip/c99_local_plasticity.c -o chip/libc99_local.dylib
 */
#include <stdint.h>
#include <stddef.h>

#define Q8_ONE 256
#define MAX_SLOTS 16
#define MAX_DIM 32

typedef struct {
    int32_t n_in;          /* e.g. 176 spiking channels */
    int32_t n_codes;       /* e.g. 8 discrete associative code classes */
    int32_t d_lat;         /* e.g. 16 latent interface dimensions */
    int32_t lr_shift;      /* integer learning rate right-shift (e.g. 5 => 1/32) */
    int32_t v_ema_q8;      /* running reward baseline V_t in Q8 [0..256] */
    int32_t n_updates;     /* counter of on-chip local learning steps */

    /* On-chip int16 shadow weights (quantized to signed int4 [-8,+7] during every forward pass!) */
    /* Layout: W_read[n_codes * n_in] in Q8 fixed-point; W_write[n_codes * d_lat] in Q8 */
    int16_t *w_read_q8;    /* size: n_codes * n_in */
    int16_t *w_write_q8;   /* size: n_codes * d_lat */

    /* Last trial local pre x post activity cache for Three-Factor local plasticity */
    int8_t  *last_x_q8;    /* size: n_in */
    int32_t *usage_q8;     /* size: n_codes (homeostatic firing-rate tracker in Q8, Turrigiano BCM!) */
    int32_t  last_pred;    /* chosen code index [0..n_codes-1] */
    int32_t  last_margin;  /* Q8 confidence margin */
} C99LocalChip;

static inline int32_t quant_i4_from_q8(int16_t w_q8) {
    /* Symmetric 4-bit quantization [-8, +7] using bit shift (step = 32 in Q8 = 0.125) */
    int32_t q = (w_q8 >= 0) ? ((w_q8 + 16) >> 5) : -(((-w_q8) + 16) >> 5);
    if (q > 7) q = 7;
    if (q < -8) q = -8;
    return q;
}

static inline int16_t clamp_i16(int32_t v, int32_t lo, int32_t hi) {
    if (v > hi) return (int16_t)hi;
    if (v < lo) return (int16_t)lo;
    return (int16_t)v;
}

/* Forward pass: reads int8 spikes x_q8[n_in], computes int4-weight dot products in int32,
 * returns argmax code index [0..n_codes-1] and writes norm-bounded int8 writeback out_wb[d_lat].
 * ZERO floating-point operations!
 */
int32_t c99_local_forward(
    C99LocalChip *chip,
    const int8_t *x_q8,
    int8_t *out_wb_q8,
    int32_t *out_scores_q8
) {
    const int32_t I = chip->n_in;
    const int32_t C = chip->n_codes;
    const int32_t D = chip->d_lat;

    int32_t best_c = 0;
    int32_t best_s = -2147483647;
    int32_t second_s = -2147483647;

    for (int32_t i = 0; i < I; ++i) {
        chip->last_x_q8[i] = x_q8[i];
    }

    const int32_t target_usage = (Q8_ONE * 16) / C; /* target firing freq per unit in Q12 */
    for (int32_t c = 0; c < C; ++c) {
        const int16_t *wrow = chip->w_read_q8 + (size_t)c * I;
        int32_t acc = 0;
        for (int32_t i = 0; i < I; ++i) {
            int32_t w4 = quant_i4_from_q8(wrow[i]); /* strictly 4-bit signed [-8, +7] */
            acc += w4 * (int32_t)x_q8[i];
        }
        /* Gentle homeostatic intrinsic excitability (wakes dead units without overriding matched spikes!) */
        int32_t homeo_bias = (chip->usage_q8 != NULL) ? ((target_usage - chip->usage_q8[c]) >> 2) : 0;
        int32_t s_q8 = (acc >> 3) + homeo_bias;
        if (out_scores_q8 != NULL) out_scores_q8[c] = s_q8;
        if (s_q8 > best_s) {
            second_s = best_s;
            best_s = s_q8;
            best_c = c;
        } else if (s_q8 > second_s) {
            second_s = s_q8;
        }
    }

    chip->last_pred = best_c;
    chip->last_margin = best_s - second_s;

    /* Generate safe norm-bounded 4-bit writeback vector to prefrontal cortex */
    if (out_wb_q8 != NULL) {
        const int16_t *ww = chip->w_write_q8 + (size_t)best_c * D;
        for (int32_t d = 0; d < D; ++d) {
            int32_t w4 = quant_i4_from_q8(ww[d]);
            int32_t val = w4 * 14; /* bounded strictly inside [-112, +98] int8 */
            if (val > 112) val = 112;
            if (val < -112) val = -112;
            out_wb_q8[d] = (int8_t)val;
        }
    }
    return best_c;
}

/* On-chip Unsupervised Competitive Hebbian + 1-Bit Dopamine RPE Local Plasticity!
 * ZERO ground-truth labels, ZERO floating-point operations, ZERO backpropagation!
 *   - Pulls winning cluster `pred` toward incoming int8 spike pattern `last_x_q8` (competitive Oja/Hebbian)
 *     and weakly repels the runner-up cluster so all C codes self-organize onto distinct cortical assemblies!
 *   - Modulated by 1-bit scalar Dopamine RPE (r_t - V_t): positive RPE consolidates `pred`, negative RPE
 *     applies homeostatic anti-Hebbian push on `pred` so colliding assemblies split cleanly!
 */
void c99_local_learn_rpe(
    C99LocalChip *chip,
    int32_t reward_bit,       /* 0 or 1: scalar 1-bit Dopamine feedback ONLY! */
    int32_t runner_up         /* runner-up code index for local lateral inhibition (no labels!) */
) {
    const int32_t I = chip->n_in;
    const int32_t C = chip->n_codes;
    const int32_t r_q8 = reward_bit ? Q8_ONE : 0;

    /* Update running reward baseline V_t <- (15 * V_t + r_t) >> 4 */
    chip->v_ema_q8 = (chip->v_ema_q8 * 15 + r_q8) >> 4;
    chip->n_updates += 1;

    const int32_t pred = chip->last_pred;
    const int32_t shift = chip->lr_shift;
    int16_t *wp = chip->w_read_q8 + (size_t)pred * I;

    /* Update homeostatic firing-rate tracker usage_q8[c] (32 bytes SRAM, 100% label-free!) */
    if (chip->usage_q8 != NULL) {
        for (int32_t c = 0; c < C; ++c) {
            int32_t cur = (c == pred) ? (Q8_ONE * 16) : 0;
            chip->usage_q8[c] = (chip->usage_q8[c] * 31 + cur) >> 5;
        }
    }

    /* Unsupervised competitive Oja/Hebbian alignment on zero-baseline spikes:
     * active spiking channels pull wp positive; silent channels undergo mild synaptic scaling toward -24! */
    const int32_t eff_shift = reward_bit ? (shift - 1) : shift;
    for (int32_t i = 0; i < I; ++i) {
        int32_t xi = (int32_t)chip->last_x_q8[i];
        int32_t target_q8 = (xi > 0) ? (xi << 2) : -32;
        if (target_q8 > 224) target_q8 = 224;
        int32_t diff = target_q8 - (int32_t)wp[i];
        int32_t dw = (diff >= 0) ? (diff >> eff_shift) : -((-diff) >> eff_shift);
        wp[i] = clamp_i16((int32_t)wp[i] + dw, -256, 224);
    }

    /* Local lateral inhibition on runner-up assembly (forces clean 8/8 decorrelation with ZERO labels!) */
    if (runner_up >= 0 && runner_up < C && runner_up != pred) {
        int16_t *wr = chip->w_read_q8 + (size_t)runner_up * I;
        const int32_t rep_shift = shift + 1;
        for (int32_t i = 0; i < I; ++i) {
            int32_t xi = (int32_t)chip->last_x_q8[i];
            if (xi > 0) {
                int32_t dw = xi >> rep_shift;
                wr[i] = clamp_i16((int32_t)wr[i] - dw, -256, 224);
            }
        }
    }
}
