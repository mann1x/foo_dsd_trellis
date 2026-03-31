/*
 * ringbuf.h — Lock-free single-producer single-consumer byte ring buffer.
 *
 * Based on PortAudio's pa_ringbuffer (Phil Burk) and JUCE's AbstractFifo.
 * Uses monotonic atomic counters (never wrap) with power-of-two capacity.
 * On x86/x64, volatile LONG + InterlockedExchange provides sufficient
 * memory ordering (TSO guarantees store-release / load-acquire semantics).
 *
 * Thread safety: exactly ONE writer thread and ONE reader thread.
 * No locks, no blocking, no system calls in the hot path.
 */
#ifndef RINGBUF_H
#define RINGBUF_H

#include <windows.h>
#include <string.h>
#include <stdlib.h>

typedef struct {
    unsigned char  *data;
    int             capacity;   /* power of 2, in bytes */
    int             mask;       /* capacity - 1 */
    /* Cache-line separated to avoid false sharing between producer/consumer */
    __declspec(align(64)) volatile LONG write_pos;  /* monotonic byte offset */
    __declspec(align(64)) volatile LONG read_pos;   /* monotonic byte offset */
} ringbuf_t;

/* Round up to next power of 2 */
static __inline int ringbuf_next_pow2(int v) {
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4;
    v |= v >> 8; v |= v >> 16;
    return v + 1;
}

/* Initialize ring buffer. capacity_hint is rounded up to power of 2. */
static __inline int ringbuf_init(ringbuf_t *rb, int capacity_hint) {
    int cap = ringbuf_next_pow2(capacity_hint);
    if (cap < 64) cap = 64;
    rb->data = (unsigned char *)malloc((size_t)cap);
    if (!rb->data) return -1;
    rb->capacity = cap;
    rb->mask = cap - 1;
    rb->write_pos = 0;
    rb->read_pos = 0;
    return 0;
}

static __inline void ringbuf_free(ringbuf_t *rb) {
    free(rb->data);
    rb->data = NULL;
    rb->capacity = 0;
}

static __inline void ringbuf_reset(ringbuf_t *rb) {
    InterlockedExchange(&rb->write_pos, 0);
    InterlockedExchange(&rb->read_pos, 0);
}

/* Bytes available for reading */
static __inline int ringbuf_available(const ringbuf_t *rb) {
    LONG w = rb->write_pos;  /* acquire: x86 volatile read is sufficient */
    LONG r = rb->read_pos;
    return (int)(w - r);
}

/* Bytes of free space for writing */
static __inline int ringbuf_free_space(const ringbuf_t *rb) {
    return rb->capacity - ringbuf_available(rb);
}

/* Write bytes into ring buffer. Returns bytes actually written (may be less if full). */
static __inline int ringbuf_write(ringbuf_t *rb, const void *src, int bytes) {
    int avail = ringbuf_free_space(rb);
    if (bytes > avail) bytes = avail;
    if (bytes <= 0) return 0;

    LONG wp = rb->write_pos;
    int idx = (int)(wp & rb->mask);
    const unsigned char *s = (const unsigned char *)src;

    /* First contiguous block (up to end of buffer) */
    int block1 = rb->capacity - idx;
    if (block1 > bytes) block1 = bytes;
    memcpy(rb->data + idx, s, (size_t)block1);

    /* Wrap-around block */
    int block2 = bytes - block1;
    if (block2 > 0)
        memcpy(rb->data, s + block1, (size_t)block2);

    /* Release: make data visible before advancing write position.
     * InterlockedExchange provides full barrier on x86 (overkill but correct). */
    InterlockedExchange(&rb->write_pos, wp + bytes);
    return bytes;
}

/* Read bytes from ring buffer. Returns bytes actually read (may be less if empty). */
static __inline int ringbuf_read(ringbuf_t *rb, void *dst, int bytes) {
    int avail = ringbuf_available(rb);
    if (bytes > avail) bytes = avail;
    if (bytes <= 0) return 0;

    LONG rp = rb->read_pos;
    int idx = (int)(rp & rb->mask);
    unsigned char *d = (unsigned char *)dst;

    int block1 = rb->capacity - idx;
    if (block1 > bytes) block1 = bytes;
    memcpy(d, rb->data + idx, (size_t)block1);

    int block2 = bytes - block1;
    if (block2 > 0)
        memcpy(d + block1, rb->data, (size_t)block2);

    InterlockedExchange(&rb->read_pos, rp + bytes);
    return bytes;
}

#endif /* RINGBUF_H */
