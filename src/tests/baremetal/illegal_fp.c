int main(void) {
    /* fadd.s f0, f0, f0: deliberately illegal on RV64IMC. */
    __asm__ volatile (".word 0x00000053");
    return 0;
}
