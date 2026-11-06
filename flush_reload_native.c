/* Timing helpers for the local, cooperative Flush+Reload verification demo.
 * Build: cc -O2 -shared -fPIC flush_reload_native.c -o /tmp/flush_reload_native.so
 */
#include <stdint.h>
#include <x86intrin.h>

void fr_flush(const void *address) {
    _mm_mfence();
    _mm_clflush(address);
    _mm_mfence();
}

void fr_touch(const void *address) {
    (void)*(const volatile uint8_t *)address;
    _mm_lfence();
}

uint64_t fr_reload(const void *address) {
    unsigned int auxiliary;
    _mm_lfence();
    uint64_t start = __rdtsc();
    _mm_lfence();
    (void)*(const volatile uint8_t *)address;
    uint64_t end = __rdtscp(&auxiliary);
    _mm_lfence();
    return end - start;
}
