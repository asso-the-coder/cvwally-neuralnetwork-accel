#ifndef INSTRET_H
#define INSTRET_H
#include <stdint.h>
static inline uint64_t read_instret(void) {
    uint64_t value;
    __asm__ volatile ("csrr %0, minstret" : "=r"(value) : : "memory");
    return value;
}
#endif
