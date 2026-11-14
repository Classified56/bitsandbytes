/* Timing only. Used exclusively with our cooperative shared activation buffer. */
#include <stdint.h>
#include <stddef.h>
#include <x86intrin.h>

void flush_lines(const uint8_t *p, size_t n) {
    _mm_mfence();
    for (size_t i=0; i<n; ++i) _mm_clflush(p+64*i);
    _mm_mfence();
}

void probe_lines(const uint8_t *p, const uint32_t *order, uint32_t *out, size_t n) {
    unsigned aux;
    for (size_t j=0; j<n; ++j) {
        size_t i=order[j];
        _mm_lfence();
        uint64_t start=__rdtsc();
        _mm_lfence();
        (void)*(const volatile uint8_t *)(p+64*i);
        uint64_t end=__rdtscp(&aux);
        _mm_lfence();
        out[i]=(uint32_t)(end-start);
    }
}
