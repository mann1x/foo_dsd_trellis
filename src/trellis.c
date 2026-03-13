/*
 * foo_dsd_trellis — Trellis (Viterbi look-ahead) SDM core
 *
 * Ported from mansr/sox sdm.c (LGPL v2.1+)
 * Adapted for float32 I/O and foobar2000 DSP context.
 */

#include "../include/trellis.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _MSC_VER
#include <immintrin.h>
#endif

#define PATH_HASH_MASK (PATH_HASH_SIZE - 1)

#define sqr(x) ((x) * (x))

/* ─── NTF state advance (hot path — force-inlined) ─── */

/*
 * Generic filter calc for arbitrary order.
 * Used as fallback when order doesn't match specialized versions.
 */
static __forceinline double sdm_filter_calc_generic(const double *s, double *d,
                                                      const ntf_filter_t *f,
                                                      double x, double y)
{
    const double *a = f->a;
    const double *g = f->g;
    double v;
    int i;

    d[0] = s[0] - g[0] * s[1] + x - y;
    v = x + a[0] * d[0];

    for (i = 1; i < f->order - 1; i++) {
        d[i] = s[i] + s[i - 1] - g[i] * s[i + 1];
        v += a[i] * d[i];
    }

    d[i] = s[i] + s[i - 1];
    v += a[i] * d[i];

    return v;
}

/* Fully unrolled order-4 filter calc */
static __forceinline double sdm_filter_calc_o4(const double *s, double *d,
                                                 const double *a, const double *g,
                                                 double x, double y)
{
    d[0] = s[0] - g[0] * s[1] + x - y;
    d[1] = s[1] + s[0] - g[1] * s[2];
    d[2] = s[2] + s[1] - g[2] * s[3];
    d[3] = s[3] + s[2];
    return x + a[0] * d[0] + a[1] * d[1] + a[2] * d[2] + a[3] * d[3];
}

/* Fully unrolled order-5 filter calc */
static __forceinline double sdm_filter_calc_o5(const double *s, double *d,
                                                 const double *a, const double *g,
                                                 double x, double y)
{
    d[0] = s[0] - g[0] * s[1] + x - y;
    d[1] = s[1] + s[0] - g[1] * s[2];
    d[2] = s[2] + s[1] - g[2] * s[3];
    d[3] = s[3] + s[2] - g[3] * s[4];
    d[4] = s[4] + s[3];
    return x + a[0] * d[0] + a[1] * d[1] + a[2] * d[2] + a[3] * d[3] + a[4] * d[4];
}

/* Fully unrolled order-6 filter calc */
static __forceinline double sdm_filter_calc_o6(const double *s, double *d,
                                                 const double *a, const double *g,
                                                 double x, double y)
{
    d[0] = s[0] - g[0] * s[1] + x - y;
    d[1] = s[1] + s[0] - g[1] * s[2];
    d[2] = s[2] + s[1] - g[2] * s[3];
    d[3] = s[3] + s[2] - g[3] * s[4];
    d[4] = s[4] + s[3] - g[4] * s[5];
    d[5] = s[5] + s[4];
    return x + a[0] * d[0] + a[1] * d[1] + a[2] * d[2]
             + a[3] * d[3] + a[4] * d[4] + a[5] * d[5];
}

/* Fully unrolled order-7 filter calc */
static __forceinline double sdm_filter_calc_o7(const double *s, double *d,
                                                 const double *a, const double *g,
                                                 double x, double y)
{
    d[0] = s[0] - g[0] * s[1] + x - y;
    d[1] = s[1] + s[0] - g[1] * s[2];
    d[2] = s[2] + s[1] - g[2] * s[3];
    d[3] = s[3] + s[2] - g[3] * s[4];
    d[4] = s[4] + s[3] - g[4] * s[5];
    d[5] = s[5] + s[4] - g[5] * s[6];
    d[6] = s[6] + s[5];
    return x + a[0] * d[0] + a[1] * d[1] + a[2] * d[2]
             + a[3] * d[3] + a[4] * d[4] + a[5] * d[5] + a[6] * d[6];
}

/* Fully unrolled order-8 filter calc */
static __forceinline double sdm_filter_calc_o8(const double *s, double *d,
                                                 const double *a, const double *g,
                                                 double x, double y)
{
    d[0] = s[0] - g[0] * s[1] + x - y;
    d[1] = s[1] + s[0] - g[1] * s[2];
    d[2] = s[2] + s[1] - g[2] * s[3];
    d[3] = s[3] + s[2] - g[3] * s[4];
    d[4] = s[4] + s[3] - g[4] * s[5];
    d[5] = s[5] + s[4] - g[5] * s[6];
    d[6] = s[6] + s[5] - g[6] * s[7];
    d[7] = s[7] + s[6];
    return x + a[0] * d[0] + a[1] * d[1] + a[2] * d[2] + a[3] * d[3]
             + a[4] * d[4] + a[5] * d[5] + a[6] * d[6] + a[7] * d[7];
}

/* Dispatch to order-specialized filter calc */
static __forceinline double sdm_filter_calc(const double *s, double *d,
                                              const ntf_filter_t *f,
                                              double x, double y)
{
    switch (f->order) {
    case 4: return sdm_filter_calc_o4(s, d, f->a, f->g, x, y);
    case 5: return sdm_filter_calc_o5(s, d, f->a, f->g, x, y);
    case 6: return sdm_filter_calc_o6(s, d, f->a, f->g, x, y);
    case 7: return sdm_filter_calc_o7(s, d, f->a, f->g, x, y);
    case 8: return sdm_filter_calc_o8(s, d, f->a, f->g, x, y);
    default: return sdm_filter_calc_generic(s, d, f, x, y);
    }
}

static __forceinline void sdm_filter_calc2(sdm_state_t *src, sdm_state_t *dst,
                                             const ntf_filter_t *f, double x)
{
    const double *a = f->a;
    double v;

    v = sdm_filter_calc(src->state, dst[0].state, f, x, 0.0);

    /* Copy state vector: dst[1] = dst[0] using AVX2 where possible */
#if defined(_MSC_VER) && defined(__AVX2__)
    {
        /* Copy first 4 doubles (32 bytes) with AVX2, rest scalar */
        __m256d st4 = _mm256_loadu_pd(dst[0].state);
        _mm256_storeu_pd(dst[1].state, st4);
        for (int i = 4; i < f->order; i++)
            dst[1].state[i] = dst[0].state[i];
    }
#else
    for (int i = 0; i < f->order; i++)
        dst[1].state[i] = dst[0].state[i];
#endif

    dst[0].state[0] += 1.0;
    dst[1].state[0] -= 1.0;

    dst[0].cost = src->cost + sqr(v + a[0]);
    dst[1].cost = src->cost + sqr(v - a[0]);
}

/* ─── History buffer management ─── */

static inline unsigned sdm_histbuf_get(sdm_context_t *p)
{
    return p->hist_free[--p->hist_fnum];
}

static inline void sdm_histbuf_put(sdm_context_t *p, unsigned h)
{
    p->hist_free[p->hist_fnum++] = (uint8_t)h;
}

/* ─── Bit-packed history access ─── */

static inline unsigned get_bit(const uint8_t *p, unsigned i)
{
    return (p[i >> 3] >> (i & 7)) & 1;
}

static inline void put_bit(uint8_t *p, unsigned i, unsigned v)
{
    int b = p[i >> 3];
    int s = i & 7;
    b &= ~(1 << s);
    b |= (int)v << s;
    p[i >> 3] = (uint8_t)b;
}

static inline unsigned sdm_hist_get(const sdm_context_t *p, unsigned h, unsigned i)
{
    return get_bit(p->hist[h], i);
}

static inline void sdm_hist_put(sdm_context_t *p, unsigned h, unsigned i, unsigned v)
{
    put_bit(p->hist[h], i, v);
}

static inline void sdm_hist_copy(sdm_context_t *p, unsigned d, unsigned s)
{
    memcpy(p->hist[d], p->hist[s], (size_t)(p->trellis_lat + 7) / 8);
}

/* ─── Cost comparison (IEEE 754 integer comparison trick) ─── */

static inline int64_t dbl2int64(double a)
{
    union { double d; int64_t i; } v;
    v.d = a;
    return v.i;
}

static __forceinline int sdm_cmplt(const sdm_state_t *a, const sdm_state_t *b)
{
    return dbl2int64(a->cost) < dbl2int64(b->cost);
}

static __forceinline int sdm_cmple(const sdm_state_t *a, const sdm_state_t *b)
{
    return dbl2int64(a->cost) <= dbl2int64(b->cost);
}

/* ─── Path deduplication via hash table ─── */

static sdm_state_t *sdm_check_path(sdm_context_t *p, sdm_state_t *s)
{
    unsigned index = s->path & PATH_HASH_MASK;
    sdm_state_t **hash = p->path_hash;
    sdm_state_t *t = hash[index];

    while (t) {
        if (t->path == s->path)
            return t;
        t = t->path_list;
    }

    s->path_list = hash[index];
    hash[index] = s;

    return NULL;
}

/* ─── Candidate sorting with path dedup ─── */

static unsigned sdm_sort_cands(sdm_context_t *p, sdm_trellis_t *st)
{
    sdm_state_t *r = NULL, *s, *t;
    sdm_state_t *min = NULL;
    unsigned i, j, n;

    for (i = 0; i < 2 * p->num_cands; i++) {
        s = &st->sdm[i];
        p->path_hash[s->path & PATH_HASH_MASK] = NULL;
        if (!min || sdm_cmplt(s, min))
            min = s;
    }

    for (i = 0, n = 0; i < 2 * p->num_cands; i++) {
        s = &st->sdm[i];

        if (s->next != min->next)
            continue;

        if (n == p->trellis_num && sdm_cmple(st->act[n - 1], s))
            continue;

        t = sdm_check_path(p, s);

        if (!t) {
            for (j = n; j > 0; j--) {
                t = st->act[j - 1];
                if (sdm_cmple(t, s))
                    break;
                st->act[j] = t;
            }
            if (j < p->trellis_num)
                st->act[j] = s;
            if (n < p->trellis_num)
                n++;
            continue;
        }

        if (sdm_cmple(t, s))
            continue;

        for (j = 0; j < n; j++) {
            r = st->act[j];
            if (sdm_cmple(s, r))
                break;
        }

        st->act[j++] = s;

        while (r != t && j < n) {
            sdm_state_t *u = st->act[j];
            st->act[j] = r;
            r = u;
            j++;
        }
    }

    return n;
}

/* ─── Per-candidate step ─── */

static __forceinline void sdm_step(sdm_context_t *p, sdm_state_t *cur,
                                    sdm_state_t *next, double x)
{
    sdm_filter_calc2(cur, next, p->filter, x);

    for (int i = 0; i < 2; i++) {
        next[i].path = (cur->path << 1 | (unsigned)i) & p->trellis_mask;
        next[i].hist = cur->hist;
        next[i].next = cur->next;
        next[i].parent = cur;
    }
}

/* ─── Main per-sample trellis algorithm ─── */

static double sdm_sample_trellis(sdm_context_t *p, double x)
{
    sdm_trellis_t *st_cur  = &p->trellis[p->idx];
    sdm_trellis_t *st_next = &p->trellis[p->idx ^ 1];
    double min_cost;
    unsigned new_cands;
    unsigned next_pos;
    unsigned output;
    unsigned i;

    next_pos = p->pos + 1;
    if (next_pos == p->trellis_lat)
        next_pos = 0;

    for (i = 0; i < p->num_cands; i++) {
        sdm_state_t *cur  = st_cur->act[i];
        sdm_state_t *next = &st_next->sdm[2 * i];
        sdm_step(p, cur, next, x);
        cur->next = (uint8_t)sdm_hist_get(p, cur->hist, next_pos);
        cur->hist_used = 0;
    }

    new_cands = sdm_sort_cands(p, st_next);
    min_cost = st_next->act[0]->cost;
    output = st_next->act[0]->next;

    for (i = 0; i < new_cands; i++) {
        sdm_state_t *s = st_next->act[i];
        if (s->parent->hist_used) {
            unsigned h = sdm_histbuf_get(p);
            sdm_hist_copy(p, h, s->hist);
            s->hist = (uint8_t)h;
        } else {
            s->parent->hist_used = 1;
        }

        s->cost -= min_cost;
        s->next = s->parent->next;
        sdm_hist_put(p, s->hist, p->pos, s->path & 1);
    }

    for (i = 0; i < p->num_cands; i++) {
        sdm_state_t *s = st_cur->act[i];
        if (!s->hist_used)
            sdm_histbuf_put(p, s->hist);
    }

    if (new_cands < p->num_cands)
        p->conv_fail++;

    p->num_cands = new_cands;
    p->pos = next_pos;
    p->idx ^= 1;

    return output ? 1.0 : -1.0;
}

/* ─── Public API ─── */

int sdm_context_init(sdm_context_t *ctx, const ntf_filter_t *filter,
                     int trellis_depth, int trellis_cands, int trellis_lat) {
    memset(ctx, 0, sizeof(*ctx));

    if (!filter)
        return -1;

    if (trellis_depth > SDM_TRELLIS_MAX_ORDER ||
        trellis_cands > SDM_TRELLIS_MAX_NUM ||
        trellis_lat > SDM_TRELLIS_MAX_LAT)
        return -1;

    if (trellis_depth < 1 || trellis_cands < 1 || trellis_lat < 1)
        return -1;

    ctx->filter = filter;
    ctx->trellis_num = (uint32_t)trellis_cands;
    ctx->trellis_lat = (uint32_t)trellis_lat;
    ctx->trellis_mask = ((uint64_t)1 << trellis_depth) - 1;
    ctx->num_cands = 1;

    /* Init history buffer free list */
    for (unsigned i = 0; i < 2u * (unsigned)trellis_cands; i++)
        ctx->hist_free[ctx->hist_fnum++] = (uint8_t)i;

    /* Init first candidate */
    sdm_trellis_t *st = &ctx->trellis[0];
    st->sdm[0].hist = (uint8_t)sdm_histbuf_get(ctx);
    st->sdm[0].path = 0;
    st->act[0] = &st->sdm[0];

    return 0;
}

size_t sdm_process_block(sdm_context_t *ctx,
                         const float *in, float *out, size_t count) {
    float *outp = out;
    size_t len = count;
    double x;

    /* Fill latency buffer first (no output produced) */
    if (ctx->pending < ctx->trellis_lat) {
        size_t pre = ctx->trellis_lat - ctx->pending;
        if (pre > len)
            pre = len;
        ctx->pending += (unsigned)pre;
        len -= pre;
        while (pre--) {
            x = (double)*in++ * 0.5;
            sdm_sample_trellis(ctx, x);
        }
    }

    /* Produce output samples */
    while (len--) {
        x = (double)*in++ * 0.5;
        *outp++ = (float)sdm_sample_trellis(ctx, x);
    }

    return (size_t)(outp - out);
}

size_t sdm_drain(sdm_context_t *ctx, float *out, size_t max_out) {
    size_t len = ctx->pending;
    if (len > max_out)
        len = max_out;

    /* If we haven't started draining yet and didn't fully fill latency,
       flush the remaining latency slots by feeding zeros */
    if (!ctx->draining && ctx->pending < ctx->trellis_lat) {
        unsigned flush = ctx->trellis_lat - ctx->pending;
        while (flush--)
            sdm_sample_trellis(ctx, 0.0);
    }

    ctx->draining = 1;
    ctx->pending -= (unsigned)len;

    for (size_t i = 0; i < len; i++)
        out[i] = (float)sdm_sample_trellis(ctx, 0.0);

    return len;
}

void sdm_context_reset(sdm_context_t *ctx) {
    if (!ctx->filter)
        return;

    const ntf_filter_t *f = ctx->filter;
    uint32_t mask = ctx->trellis_mask;
    uint32_t num = ctx->trellis_num;
    uint32_t lat = ctx->trellis_lat;

    memset(ctx->trellis, 0, sizeof(ctx->trellis));
    memset(ctx->path_hash, 0, sizeof(ctx->path_hash));
    memset(ctx->hist, 0, sizeof(ctx->hist));

    ctx->hist_fnum = 0;
    ctx->num_cands = 1;
    ctx->pos = 0;
    ctx->pending = 0;
    ctx->draining = 0;
    ctx->idx = 0;
    ctx->prev_y = 0.0;
    ctx->conv_fail = 0;

    ctx->filter = f;
    ctx->trellis_mask = mask;
    ctx->trellis_num = num;
    ctx->trellis_lat = lat;

    for (unsigned i = 0; i < 2u * num; i++)
        ctx->hist_free[ctx->hist_fnum++] = (uint8_t)i;

    sdm_trellis_t *st = &ctx->trellis[0];
    st->sdm[0].hist = (uint8_t)sdm_histbuf_get(ctx);
    st->sdm[0].path = 0;
    st->act[0] = &st->sdm[0];
}

void sdm_context_free(sdm_context_t *ctx) {
    memset(ctx, 0, sizeof(*ctx));
}
