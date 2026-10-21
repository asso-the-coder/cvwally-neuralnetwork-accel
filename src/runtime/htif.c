#include "htif.h"

volatile uint64_t tohost __attribute__((section(".tohost"), aligned(64)));
volatile uint64_t fromhost __attribute__((section(".tohost"), aligned(64)));

void htif_putchar(int ch) {
    while (tohost != 0) {}
    tohost = (UINT64_C(1) << 56) | (UINT64_C(1) << 48) | (uint8_t)ch;
    /* Console writes have no fromhost reply; consumption clears tohost. */
    while (tohost != 0) {}
}

_Noreturn void htif_exit(int code) {
    while (tohost != 0) {}
    tohost = ((uint64_t)(unsigned int)code << 1) | 1;
    for (;;) {}
}

static void putstr(const char *s) {
    while (*s) htif_putchar(*s++);
}

static void puthex(uint64_t value) {
    const char digits[] = "0123456789abcdef";
    putstr("0x");
    for (int shift = 60; shift >= 0; shift -= 4)
        htif_putchar(digits[(value >> shift) & 15]);
}

_Noreturn void htif_trap(uint64_t cause, uint64_t epc, uint64_t tval) {
    /* Avoid libc state when reporting a fatal exception. */
    putstr("trap: mcause="); puthex(cause);
    putstr(" mepc="); puthex(epc);
    putstr(" mtval="); puthex(tval);
    htif_putchar('\n');
    htif_exit(1);
}
