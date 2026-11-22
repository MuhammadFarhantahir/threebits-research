/* Threebits RP2350 / ARM Cortex-M33 USB-CDC Wire Protocol (C99, zero floats, zero malloc).
 *
 * Shared between Raspberry Pi Pico 2 firmware (main.c) and host tests / level3_app/server.py.
 * Line-framed ASCII + CRC-16/CCITT-FALSE so USB-CDC never loses framing and every reply is
 * verified end-to-end across hardware UART/USB-CDC or POSIX PTY loopback.
 */
#ifndef THREEBITS_RP2350_PROTOCOL_H
#define THREEBITS_RP2350_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

#define TB_PROTO_VERSION 1
#define TB_LINE_MAX      768
#define TB_LOCAL_IN      176
#define TB_LOCAL_CODES   8
#define TB_LOCAL_DLAT    16

/* CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) over ASCII payload prefix before " crc=" */
static inline uint16_t tb_crc16_ccitt(const uint8_t *data, size_t len) {
    uint16_t crc = 0xFFFFu;
    for (size_t i = 0; i < len; ++i) {
        crc ^= (uint16_t)((uint16_t)data[i] << 8);
        for (int b = 0; b < 8; ++b) {
            if (crc & 0x8000u) {
                crc = (uint16_t)((crc << 1) ^ 0x1021u);
            } else {
                crc = (uint16_t)(crc << 1);
            }
        }
    }
    return crc;
}

static inline int tb_hex_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/* Decode `n_bytes` signed int8 values from `2 * n_bytes` hex characters. Returns 0 on success. */
static inline int tb_hex_to_i8(const char *hex, int8_t *out, size_t n_bytes) {
    for (size_t i = 0; i < n_bytes; ++i) {
        int hi = tb_hex_nibble(hex[2 * i]);
        int lo = tb_hex_nibble(hex[2 * i + 1]);
        if (hi < 0 || lo < 0) return -1;
        out[i] = (int8_t)(uint8_t)((hi << 4) | lo);
    }
    return 0;
}

/* Encode `n_bytes` signed int8 values into `2 * n_bytes` lowercase hex chars + NUL terminator. */
static inline void tb_i8_to_hex(const int8_t *in, size_t n_bytes, char *hex_out) {
    static const char HEX[] = "0123456789abcdef";
    for (size_t i = 0; i < n_bytes; ++i) {
        uint8_t b = (uint8_t)in[i];
        hex_out[2 * i] = HEX[(b >> 4) & 0x0Fu];
        hex_out[2 * i + 1] = HEX[b & 0x0Fu];
    }
    hex_out[2 * n_bytes] = '\0';
}

#endif /* THREEBITS_RP2350_PROTOCOL_H */
