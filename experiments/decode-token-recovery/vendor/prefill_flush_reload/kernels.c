/* Synthetic down-projection kernels. Only gather() reads original FP16
 * activations inside the cache observation window. All buffers are supplied
 * by Python and allocated before measuring; this file performs no allocation.
 */
#include <cpuid.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <x86intrin.h>

/* Check before any conversion/timing instruction can run. F16C has AVX
 * OS-state requirements, which the compiler's feature detection checks.
 */
int platform_supported(void) {
    unsigned int a, b, c, d;
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("f16c")) return 0;
    if (!__get_cpuid(0x80000001, &a, &b, &c, &d) || !(d & (1u << 27))) return 0;
    if (!__get_cpuid(1, &a, &b, &c, &d) || ((b >> 8) & 255) * 8 != 64) return 0;
    return 1;
}

void fr_flush(const void *address);
uint64_t fr_reload(const void *address);

/* Explicit IEEE binary16 conversions via this CPU's F16C instructions. */
static float from_half(uint16_t value) { return _cvtsh_ss(value); }
static uint16_t to_half(float value) {
    return _cvtss_sh(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

static uint32_t step(uint32_t *state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

/* Generate bounded, synthetic post-FFN activations and FP16-rounded weights.
 * No trained-model activation distribution or SwiGLU execution is claimed.
 */
void fill_half(uint16_t *buffer, size_t count, uint32_t seed, float bound) {
    uint32_t state = seed ? seed : 1;
    for (size_t i = 0; i < count; ++i)
        buffer[i] = to_half(((float)(step(&state) % 20001) / 10000.0f - 1.0f) * bound);
}

void inject(uint16_t *a, size_t tokens, size_t width,
            const size_t *columns, size_t count, float value) {
    for (size_t j = 0; j < count; ++j)
        a[((j * 17) % tokens) * width + columns[j]] = to_half(j % 2 ? -value : value);
}

/* Match the relevant default-backend policy: detect a batch-wide column
 * union, compute each row's scale excluding threshold-crossing values,
 * quantize, and suppress the complete selected columns in the INT8 result.
 * The original FP16 buffer remains intact for the later gather.
 */
size_t quantize(const uint16_t *a, int8_t *q, float *scales,
                size_t *columns, uint8_t *mask, size_t tokens, size_t width,
                float threshold) {
    for (size_t k = 0; k < width; ++k) mask[k] = 0;
    for (size_t r = 0; r < tokens; ++r) {
        float maximum = 0;
        for (size_t k = 0; k < width; ++k) {
            float value = fabsf(from_half(a[r * width + k]));
            if (threshold > 0 && value >= threshold) mask[k] = 1;
            else if (value > maximum) maximum = value;
        }
        scales[r] = maximum;
    }
    size_t count = 0;
    for (size_t k = 0; k < width; ++k) if (mask[k]) columns[count++] = k;
    for (size_t r = 0; r < tokens; ++r)
        for (size_t k = 0; k < width; ++k) {
            float value = from_half(a[r * width + k]);
            q[r * width + k] = mask[k] || scales[r] == 0 ? 0 :
                (int8_t)nearbyintf(value * (127.0f / scales[r]));
        }
    return count;
}

/* Preserve FP16 bits during extraction. Volatile reads prevent the compiler
 * from replacing sparse accesses with wider reads of non-selected columns.
 * Destination is a separate prefaulted buffer, just as subA is a copy.
 */
void gather(const volatile uint16_t *a, uint16_t *sub,
            const size_t *columns, size_t count, size_t tokens, size_t width) {
    for (size_t r = 0; r < tokens; ++r)
        for (size_t j = 0; j < count; ++j)
            sub[r * count + j] = a[r * width + columns[j]];
    _mm_lfence();
}

/* Operate once per physical 64-byte line. Random probe order is supplied
 * by Python before flushing, to reduce sequential prefetch contamination.
 */
void flush_lines(const uint8_t *a, size_t lines) {
    _mm_mfence();
    for (size_t i = 0; i < lines; ++i) _mm_clflush(a + 64 * i);
    _mm_mfence();
}

void probe_lines(const uint8_t *a, const size_t *order, uint64_t *times, size_t lines) {
    for (size_t i = 0; i < lines; ++i) {
        size_t line = order[i];
        times[line] = fr_reload(a + 64 * line);
    }
}

/* Expand only after observation. Outlier weights are reconstructed from
 * INT8 + row scales and rounded to FP16, rather than fetched from original W.
 */
void expand_weights(const uint16_t *original, const int8_t *q, const float *scales,
                    float *fp, float *dequant, size_t hidden, size_t width) {
    for (size_t h = 0; h < hidden; ++h)
        for (size_t k = 0; k < width; ++k) {
            size_t i = h * width + k;
            fp[i] = from_half(original[i]);
            dequant[i] = from_half(to_half((float)q[i] * scales[h] / 127.0f));
        }
}

/* Actual INT8 products with INT32 accumulation (a synthetic arithmetic
 * choice; this machine's bitsandbytes backend instead has an FP32 fallback).
 * Dequantize normal output to FP16, then add the gathered correction with
 * FP32 accumulation and round the final result to FP16.
 */
void down_project(const int8_t *qa, const int8_t *qw, const float *sa, const float *sw,
                  const uint16_t *sub, const size_t *columns, size_t count,
                  const float *dequant_w, float *normal, float *mixed,
                  size_t tokens, size_t width, size_t hidden) {
    for (size_t r = 0; r < tokens; ++r)
        for (size_t h = 0; h < hidden; ++h) {
            int32_t sum = 0;
            for (size_t k = 0; k < width; ++k)
                sum += (int32_t)qa[r * width + k] * qw[h * width + k];
            float base = from_half(to_half((float)sum * (sa[r] * sw[h]) / (127.0f * 127.0f)));
            float correction = 0;
            for (size_t j = 0; j < count; ++j)
                correction += from_half(sub[r * count + j]) * dequant_w[h * width + columns[j]];
            normal[r * hidden + h] = base;
            mixed[r * hidden + h] = from_half(to_half(base + correction));
        }
}

/* Two FP32 references distinguish activation/arithmetic error from weight
 * quantization error. Both use the full, original FP16 activation values.
 */
void references(const uint16_t *a, const float *original_w, const float *dequant_w,
                float *original_ref, float *dequant_ref,
                size_t tokens, size_t width, size_t hidden) {
    for (size_t r = 0; r < tokens; ++r)
        for (size_t h = 0; h < hidden; ++h) {
            float original = 0, reconstructed = 0;
            for (size_t k = 0; k < width; ++k) {
                float value = from_half(a[r * width + k]);
                original += value * original_w[h * width + k];
                reconstructed += value * dequant_w[h * width + k];
            }
            original_ref[r * hidden + h] = original;
            dequant_ref[r * hidden + h] = reconstructed;
        }
}
