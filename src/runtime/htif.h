#ifndef HTIF_H
#define HTIF_H
#include <stdint.h>
void htif_putchar(int ch);
_Noreturn void htif_exit(int code);
_Noreturn void htif_trap(uint64_t cause, uint64_t epc, uint64_t tval);
#endif
