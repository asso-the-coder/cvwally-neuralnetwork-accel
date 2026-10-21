OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY { ram (rwx) : ORIGIN = 0x80000000, LENGTH = 128M }
/* Reserve 64 MiB for newlib/the future arena and 64 KiB for the stack. */
__heap_size = 64M;
__stack_size = 64K;
PHDRS
{
    code PT_LOAD FLAGS(5);
    data PT_LOAD FLAGS(6);
}
SECTIONS
{
    .text : { KEEP(*(.text.init)) *(.text .text.*) } > ram :code
    .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > ram :code
    .tohost ALIGN(64) : { KEEP(*(.tohost)) } > ram :data
    .data : {
        *(.data .data.*)
        . = ALIGN(8);
        __global_pointer$ = . + 0x800;
        *(.sdata .sdata.*)
    } > ram :data
    .bss (NOLOAD) : {
        . = ALIGN(8);
        __bss_start = .;
        *(.sbss .sbss.* .bss .bss.*) *(COMMON)
        . = ALIGN(8);
        __bss_end = .;
    } > ram :data
    .heap (NOLOAD) : {
        . = ALIGN(16);
        __heap_start = .;
        . += __heap_size;
        __heap_end = .;
    } > ram :data
    .stack (NOLOAD) : {
        . = ALIGN(16);
        __stack_bottom = .;
        . += __stack_size;
        __stack_top = .;
    } > ram :data
}
